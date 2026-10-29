"""Benign rewrites and a detector-targeted lexical attack.

Synonym replacement draws a WordNet neighbor at random.
Back-translation is English to Chinese and back, with Helsinki-NLP opus-mt.
The malignant lexical attack is the TextFooler loop (Jin et al., 2020):
rank words by the drop in true-class probability when the word is deleted,
then substitute a synonym that lowers that probability, and stop when the
prediction flips. FGSM is the embedding-space counterpart, used so the
lexical block can be frozen and checked.
"""

from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn.functional as F
from nltk import pos_tag
from nltk.corpus import wordnet as wn

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "as", "by", "at", "from", "that",
    "this", "it", "its", "if", "not", "but", "we", "you", "they", "he", "she",
}


def _penn_to_wordnet(tag: str):
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("R"):
        return wn.ADV
    return None


def _synonyms(word: str, pos) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            key = name.lower()
            if " " in name or key == word.lower() or not name.isalpha() or key in seen:
                continue
            seen.add(key)
            found.append(name)
    return found


def synonym_replace(text: str, rng: np.random.Generator, rate: float = 0.15) -> str:
    """Replace about ``rate`` of eligible tokens with a random WordNet synonym."""
    words = text.split()
    if not words:
        return text
    tags = pos_tag(words)
    out = []
    for word, tag in tags:
        match = re.match(r"^([A-Za-z]+)(.*)$", word)
        if match is None:
            out.append(word)
            continue
        core, tail = match.group(1), match.group(2)
        pos = _penn_to_wordnet(tag)
        eligible = pos is not None and len(core) > 3 and core.lower() not in STOP
        if eligible and rng.random() < rate:
            choices = _synonyms(core, pos)
            if choices:
                choice = choices[int(rng.integers(0, len(choices)))]
                if core[0].isupper():
                    choice = choice[0].upper() + choice[1:]
                out.append(choice + tail)
                continue
        out.append(word)
    return " ".join(out)


def back_translate(texts: list[str], batch_size: int = 8, max_length: int = 160) -> list[str]:
    """English -> Chinese -> English. Models are loaded one at a time."""
    from transformers import MarianMTModel, MarianTokenizer

    def run(model_name: str, batch: list[str]) -> list[str]:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()
        decoded: list[str] = []
        for start in range(0, len(batch), batch_size):
            chunk = batch[start : start + batch_size]
            encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            with torch.no_grad():
                tokens = model.generate(**encoded, max_length=max_length, num_beams=2)
            decoded.extend(tokenizer.batch_decode(tokens, skip_special_tokens=True))
        del model
        return decoded

    chinese = run("Helsinki-NLP/opus-mt-en-zh", texts)
    return run("Helsinki-NLP/opus-mt-zh-en", chinese)


def textfooler(
    texts: list[str],
    labels: np.ndarray,
    predict,
    seed: int = 0,
    max_changes: int = 6,
    synonyms_per_word: int = 5,
    max_positions: int = 12,
) -> tuple[list[str], np.ndarray]:
    """Targeted synonym attack. ``predict(texts)`` returns probabilities (n, 2).

    Returns the attacked strings and a boolean array of prediction flips
    against the original labels.
    """
    rng = np.random.default_rng(seed)
    attacked: list[str] = []
    flipped = np.zeros(len(texts), dtype=bool)
    for i, (text, label) in enumerate(zip(texts, labels)):
        words = text.split()
        label = int(label)
        base = predict([text])[0]
        if int(np.argmax(base)) != label:
            attacked.append(text)
            continue
        tags = pos_tag(words) if words else []
        positions = []
        for j, (word, tag) in enumerate(tags):
            core = re.match(r"^([A-Za-z]+)", word)
            if core is None:
                continue
            token = core.group(1)
            if _penn_to_wordnet(tag) is None or len(token) <= 3 or token.lower() in STOP:
                continue
            positions.append(j)
        if not positions:
            attacked.append(text)
            continue
        deletions = [" ".join(words[:j] + words[j + 1 :]) for j in positions]
        deleted_p = predict(deletions)[:, label]
        importance = base[label] - deleted_p
        order = np.argsort(-importance)[:max_positions]
        current = words[:]
        current_p = float(base[label])
        still_correct = True
        changes = 0
        for rank in order:
            if changes >= max_changes or not still_correct:
                break
            j = positions[int(rank)]
            if j >= len(current):
                continue
            match = re.match(r"^([A-Za-z]+)(.*)$", current[j])
            if match is None:
                continue
            core, tail = match.group(1), match.group(2)
            pos = _penn_to_wordnet(tags[j][1]) if j < len(tags) else None
            choices = _synonyms(core, pos) if pos is not None else []
            if not choices:
                continue
            rng.shuffle(choices)
            choices = choices[:synonyms_per_word]
            variants = []
            for choice in choices:
                if core[0].isupper():
                    choice = choice[0].upper() + choice[1:]
                trial = current[:]
                trial[j] = choice + tail
                variants.append(trial)
            probs = predict([" ".join(trial) for trial in variants])
            true_p = probs[:, label]
            best = int(np.argmin(true_p))
            if float(true_p[best]) < current_p:
                current = variants[best]
                current_p = float(true_p[best])
                changes += 1
                still_correct = int(np.argmax(probs[best])) == label
        final = " ".join(current)
        flipped[i] = not still_correct
        attacked.append(final)
    return attacked, flipped


def _batch_block_distance(clean_hidden, pert_hidden, mask: torch.Tensor) -> tuple[float, float]:
    token_dist = 1.0 - F.cosine_similarity(clean_hidden[0], pert_hidden[0], dim=-1)
    weight = mask.float()
    lexical = float((token_dist * weight).sum() / weight.sum().clamp(min=1.0))
    semantic = float((1.0 - F.cosine_similarity(clean_hidden[-1][:, 0], pert_hidden[-1][:, 0], dim=-1)).mean())
    return lexical, semantic


def fgsm_embeddings(model, tokenizer, texts: list[str], labels: np.ndarray, eps: float, batch_size: int = 8, max_length: int = 128):
    """One-step signed gradient on input embeddings.

    Returns true-class probabilities of the perturbed embeddings, the
    perturbed tensors, their masks, and the lexical/semantic cosine distances
    from the clean embedding forward pass.
    """
    model.eval()
    all_probs = []
    all_embeds = []
    all_masks = []
    lexical_acc = 0.0
    semantic_acc = 0.0
    seen = 0
    labels = np.asarray(labels, dtype=int)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        y = torch.tensor(labels[start : start + batch_size], dtype=torch.long)
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        clean = model.get_input_embeddings()(input_ids).detach()
        pert = clean.clone().requires_grad_(True)
        logits = model(inputs_embeds=pert, attention_mask=mask).logits
        loss = F.cross_entropy(logits, y)
        loss.backward()
        stepped = (clean + eps * pert.grad.detach().sign()).detach()
        with torch.no_grad():
            clean_out = model(inputs_embeds=clean, attention_mask=mask, output_hidden_states=True)
            pert_out = model(inputs_embeds=stepped, attention_mask=mask, output_hidden_states=True)
            probs = torch.softmax(pert_out.logits, dim=-1)
        lexical, semantic = _batch_block_distance(clean_out.hidden_states, pert_out.hidden_states, mask)
        lexical_acc += lexical * len(chunk)
        semantic_acc += semantic * len(chunk)
        seen += len(chunk)
        all_probs.append(probs.cpu().numpy())
        all_embeds.append(stepped.cpu())
        all_masks.append(mask.cpu())
        model.zero_grad(set_to_none=True)
    distances = {
        "lexical": lexical_acc / max(seen, 1),
        "semantic": semantic_acc / max(seen, 1),
    }
    return np.concatenate(all_probs, axis=0), all_embeds, all_masks, distances


def score_perturbed_embeddings(model, embeds, masks) -> np.ndarray:
    """Probabilities of a new model on embeddings attacked against the old one."""
    model.eval()
    chunks = []
    for emb, mask in zip(embeds, masks):
        with torch.no_grad():
            logits = model(inputs_embeds=emb, attention_mask=mask).logits
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def choose_fgsm_eps(model, tokenizer, texts: list[str], labels: np.ndarray, candidates: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)) -> tuple[float, float]:
    """Pick the step size whose flip rate on this slice is closest to one half."""
    best_eps = candidates[0]
    best_rate = 1.0
    best_gap = 1.0
    labels = np.asarray(labels, dtype=int)
    for eps in candidates:
        probs, _, _, _ = fgsm_embeddings(model, tokenizer, texts, labels, eps)
        rate = float(np.mean(np.argmax(probs, axis=1) != labels))
        gap = abs(rate - 0.5)
        if gap < best_gap:
            best_gap = gap
            best_eps = eps
            best_rate = rate
    return best_eps, best_rate
