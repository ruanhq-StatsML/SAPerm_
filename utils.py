"""Shared feature extraction and one call to the existing benchmark."""

import wave
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def media_bytes(cell):
    # Hugging Face parquet media cells are usually {bytes, path}.
    if isinstance(cell, dict):
        cell = cell.get("bytes") or cell.get("path")
    return bytes(cell) if isinstance(cell, (bytes, bytearray, memoryview)) else Path(cell).read_bytes()


def text_meta(texts):
    import spacy
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    vader = SentimentIntensityAnalyzer()
    neg = {"no", "not", "n't", "never", "neither", "nor", "nobody", "nothing", "nowhere", "none"}
    clause = {"advcl", "ccomp", "xcomp", "relcl", "acl", "csubj", "csubjpass"}
    hedge = {"perhaps", "maybe", "possibly", "probably", "apparently", "seem", "somewhat"}
    contrast = {"but", "however", "although", "though", "nevertheless", "whereas", "yet"}
    rows = []
    for doc in nlp.pipe(texts):
        s = doc.text
        nc = max(len(s), 1)
        tokens = [t for t in doc if not t.is_space]
        words = [t for t in tokens if not t.is_punct]
        nt, ns = max(len(tokens), 1), max(len(list(doc.sents)), 1)
        freq = Counter(t.lower_ for t in words if t.is_alpha)
        depths = [sum(1 for _ in t.ancestors) for t in words]
        distances = [abs(t.i - t.head.i) for t in words if t.head != t]
        v = vader.polarity_scores(s)
        sentiment = [vader.polarity_scores(sent.text)["compound"] for sent in doc.sents]
        rows.append({
            "char_len": len(s), "word_count": len(s.split()),
            "space_ratio": s.count(" ") / nc,
            "digit_ratio": sum(c.isdigit() for c in s) / nc,
            "upper_ratio": sum(c.isupper() for c in s) / nc,
            "token_count": len(tokens), "sent_count": len(list(doc.sents)),
            "avg_token_len": sum(len(t.text) for t in tokens) / nt,
            "avg_sent_len": len(tokens) / ns,
            "type_token_ratio": len(freq) / max(sum(freq.values()), 1),
            "hapax_ratio": sum(c == 1 for c in freq.values()) / max(len(freq), 1),
            "punct_ratio": sum(t.is_punct for t in tokens) / nt,
            "exclam_ratio": sum(t.text == "!" for t in tokens) / nt,
            "question_ratio": sum(t.text == "?" for t in tokens) / nt,
            "noun_ratio": sum(t.pos_ == "NOUN" for t in tokens) / nt,
            "verb_ratio": sum(t.pos_ == "VERB" for t in tokens) / nt,
            "entity_density": len(doc.ents) / nt,
            "avg_dep_depth": np.mean(depths) if depths else 0.0,
            "max_dep_depth": max(depths, default=0),
            "avg_dep_distance": np.mean(distances) if distances else 0.0,
            "neg_ratio": sum(t.dep_ == "neg" or t.lower_ in neg for t in tokens) / nt,
            "clause_ratio": sum(t.dep_ in clause for t in tokens) / nt,
            "hedge_ratio": sum(t.lower_ in hedge for t in tokens) / nt,
            "contrast_ratio": sum(t.lower_ in contrast for t in tokens) / nt,
            "modal_ratio": sum(t.tag_ == "MD" for t in tokens) / nt,
            **{f"vader_{key}": value for key, value in v.items()},
            "polar_gap": v["pos"] - v["neg"],
            "sentiment_std": np.std(sentiment) if sentiment else 0.0,
        })
    return pd.DataFrame(rows, dtype=float)


def image_meta(bgr):
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    return {"brightness": gray.mean() / 255, "contrast": gray.std() / 255,
            "edge_density": (cv2.Canny(gray, 80, 160) > 0).mean(),
            "sat_mean": sat.mean() / 255, "hue_std": hue.std() / 179,
            "warm_ratio": ((hue < 25) | (hue > 160)).mean(),
            "laplacian_var": cv2.Laplacian(gray, cv2.CV_64F).var() / 1000}


def video_meta(path, n_frames=8):
    import cv2

    cap = cv2.VideoCapture(str(path))
    rows = []
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.unique(np.linspace(0, max(n - 1, 0), n_frames, dtype=int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                rows.append(image_meta(frame))
    finally:
        cap.release()
    if not rows:
        raise ValueError(f"No readable video frames: {path}")
    return pd.DataFrame(rows).mean().to_dict()


def audio_meta(path):
    with wave.open(str(path), "rb") as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"16-bit PCM WAV required: {path}")
    y = np.frombuffer(raw, dtype="<i2").astype(float).reshape(-1, nch).mean(axis=1) / 32768
    if len(y) < 8:
        raise ValueError(f"Empty or too-short audio: {path}")
    rms = np.sqrt(np.mean(y ** 2))
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freq = np.fft.rfftfreq(len(y), 1 / sr)
    mass = max(spec.sum(), 1e-12)
    return {"rms_energy": rms, "zcr": np.mean(np.abs(np.diff(np.sign(y)))) / 2,
            "lowband_ratio": spec[freq <= 500].sum() / mass,
            "silence_ratio": np.mean(np.abs(y) < 0.02),
            "spectral_centroid": (freq * spec).sum() / mass / (sr / 2),
            "peak_to_rms": min(np.max(np.abs(y)) / max(rms, 1e-12), 20) / 20}

#Extracting the embeddings:
def extract_lco_omni_embeddings(inputs,
    model_id = "LCO_OMNI_MODEL_ID", batch_size = 16):
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        kwargs = {'torch_dtype': dtype,
            'attn_implementation': 'sdpa',
            'low_cpu_mem_usage': True}
    else:
        dtype = torch.float16  
        kwargs = {'torch_dtype': dtype,
            'low_cpu_mem_usage': True}
    model = SentenceTransformer(model_id, 
        model_kwargs = kwargs)
    emb = model.encode(list(inputs),
        batch_size = int(batch_size),
        show_progress_bar = True,
        convert_to_numpy = True)
    return np.asarray(emb, dtype = float)

#pooling configurations/options:
def pool_operation(last_hidden, 
    attention_mask, pool_mechanism):
    if pool_mechanism == 'cls':
        return last_hidden[:, 0]
    if pool_mechanism == 'last':
        lengths = attention_mask.sum(dim = 1) - 1
        batch = torch.arange(last_hidden.size(0),
            device = last_hidden.device)
        return last_hidden[batch, lengths]
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim = 1)
    denom = mask.sum(dim = 1).clamp(min = 1e-5)
    return summed/denom
    

def extract_text_embeddings(texts,
    model_id = 'MiniLM_L6',
    config = None):
    tokenizer = AutoTokenizer.from_pretrained(cfg.name, 
        trust_remote_code = cfg.trust_remote_code)
    model = AutoModel.from_pretrained(cfg.name,
        trust_remote_code = cfg.trust_remote_code).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    chunks = []
    with torch.no_grad():
        for i in range(0, len(processed), cfg.batch_size):
            batch = processed[i:(i+cfg.batch_size)]
            inputs = tokenizer(
                batch, padding = True,
                truncation = True,
                max_length = cfg.max_length,
                return_tensors = 'pt').to(device)
            out = model(**inputs)
            pooled = pool_operation(out.last_hidden_state,
                inputs['attention_mask'],
                cfg.pooling)
            if cfg.normalize:
                pooled = torch.nn.functional.normalize(pooled, p = 2, dim = 1)
            chunks.append(pooled.cpu().numpy())
    return np.vstack(chunks)

def extract_video_embeddings(
    video_paths: Sequence[str],
    model_name = 'google/vivit-b-16x2-kinetics400'
    num_frames: int = 32,
) -> np.ndarray:
    """ViViT CLS/mean embedding for each video path (decord + transformers)."""
    import torch
    from decord import VideoReader, cpu
    from transformers import VivitImageProcessor, VivitModel

    device = _device()
    image_processor = VivitImageProcessor.from_pretrained(model_name)
    model = VivitModel.from_pretrained(model_name).to(device)
    model.eval()
    embeds = []
    for path in video_paths:
        vr = VideoReader(path, ctx=cpu(0))
        indices = np.linspace(0, max(len(vr) - 1, 0), num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        pixel_values = image_processor(list(frames), return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            out = model(pixel_values)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        embeds.append(emb)
    return np.stack(embeds, axis=0)


def encode_modalities(inputs, batch_size=8):
    return {name: np.asarray(extract_lco_omni_embeddings(values, batch_size=batch_size), dtype=float)
            for name, values in inputs.items()}

#Concatenate the embeddings through the blocks(the feature indices after PCA)
def build_X(embeddings, meta, n_pc=30):
    X, indices = pd.DataFrame(), {}
    if embeddings and meta and embeddings.keys() != meta.keys():
        raise ValueError("Embedding and meta modality keys differ")
    for name in embeddings or meta:
        block = pd.DataFrame()
        if embeddings:
            z = PCA(n_components=n_pc, random_state=0).fit_transform(StandardScaler().fit_transform(embeddings[name]))
            block = pd.DataFrame(z, columns=[f"pc_{i + 1:03d}" for i in range(n_pc)])
        if meta:
            block = pd.concat([block, pd.DataFrame(meta[name]).reset_index(drop=True)], axis=1)
        indices[name] = list(range(X.shape[1], X.shape[1] + block.shape[1]))
        X = pd.concat([X, block.add_prefix(f"{name}_")], axis=1)
    if X.empty or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Empty, mis-sized or nonfinite feature matrix")
    return X, indices

def build_meta_matrix(texts, images, audios):
    rows = []
    names = list(TEXT_NAMES) + list(IMAGE_NAMES) + list(AUDIO_NAMES)
    for i, (t, im) in zip(texts, images):
        d = extract_text_meta_features(texts)
        d.update(extract_image_meta_features(images))
        d.update(extract_audio_meta_features(audios))
        rows.append([float(d[n]) for n in names])
    return np.asarray(rows, dtype = float), names




def load_msrvtt(config = 'train_9k',
    split = 'train', max_rows = 1000):
    ds = load_dataset('friedrichor/MSR-VTT', config, split = split)
    if max_rows is not None:
        ds = ds.select(range(min(int(max_rows), len(ds))))
    return ds


def modasense_data_digestion(path, feature_extractor, Y_col, n_max = 1000):
    df = pd.read_parquet(path).iloc[:n_max].reset_index(drop = True)
    blocks = []
    indices = {}
    offset = 0
    for modality, (col, extractors) in feature_extractor.items():
        features = pd.DataFrame(
            df[col].map(extractor).tolist()).add_prefix(f"{modality}_")
        p = features.shape[1]
        indices[modality] = list(range(offset, offset + p))
        blocks.append(features)
        offset += p 
    X = pd.concat(blocks, axis = 1)
    Y = df[Y_col].str.lower().isin(
        {'fear', 'sad', 'angry', 'disgust'}).astype(int).to_numpy()
    return X, Y, indices



def get_embedding(texts, model_name = 'harrier-0.6b', MODEL_CONFIGS = MODEL_CONFIGS):
    config = MODEL_CONFIGS[model_name]
    n_dim = config.dim
    max_length = config.max_length
    batch_size = config.batch_size
    #putting them together:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = config.name
    instruction = config.default_instruction
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code = True)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    #The processed texts here:
    processed_texts = []
    if config.prefix_format:
        default_instruction = 'Given a web search query, retrieve relevant passages'
        task = instruction if instruction is not None else default_instruction
        processed_texts = [config.prefix_format.format(task = task, text = text) for text in texts]
    else:
        processed_texts = texts
    #batch setting:
    all_embeddings = []
    for i in tqdm(range(0, len(processed_texts), batch_size), desc = f'Embedding {model_name}'):
        batch_texts = processed_texts[i:(i + batch_size)]
        inputs = tokenizer(
            batch_texts,
            padding = True,
            truncation = True,
            max_length = max_length,
            return_tensors = 'pt'
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states = True)
            hidden_states = outputs.hidden_states[-1]
        #Pooling:
        pooling = config.pooling
        if pooling == 'mean':
            attention_mask = inputs['attention_mask']
            masked_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_emb = torch.sum(hidden_states * masked_expanded, dim = 1)
            mean_emb = sum_emb/torch.clamp(masked_expanded.sum(dim = 1),
                min = 1e-9)
            batch_embeds = mean_emb
        elif pooling == 'CLS':
            batch_embeds = hidden_states[:, 0, :]
        elif pooling == 'last':
            attention_mask = inputs['attention_mask']#(Batch_Size, n_tokens)
            last_token_indices = attention_mask.sum(dim = 1) - 1
            batch_embeds = hidden_states[torch.arange(hidden_states.size(0),
             device = device), last_token_indices]#
        else:
            raise ValueError(f"Invalid Pooling: {pooling}")
        all_embeddings.append(batch_embeds.to(torch.float32).cpu().numpy())
    return np.concatenate(all_embeddings)

def get_embedding_wrapper(df_csv, text_col, MODEL_CONFIGS = MODEL_CONFIGS, delimiter = None, header = None):
    if delimiter is None:
        df = pd.read_csv(df_csv)
    else:
        df = pd.read_csv(df_csv, delimiter = delimiter, on_bad_lines = 'skip', header = header)
    dataset_name = str(str(df_csv).split('.')[0])
    for i, model_name in enumerate(list(MODEL_CONFIGS.keys())):
        embedding0 = get_embedding(df[text_col].astype(str).tolist(), model_name = model_name)
        np.save(f"{dataset_name}_{model_name}_embedding.npy", embedding0)

