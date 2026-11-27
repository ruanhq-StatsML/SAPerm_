"""Back-translation (en -> de -> en) augmentation, ported from notebook cells 19-29.

Inputs are read from colab_work/translation_version/translation_version/, the folder the
notebook used on Google Drive. Copy df_corpus_trt_aug.csv, magazineSubscription_df.csv
and Digital_Music_df.csv into that folder before running. The notebook reads Digital_Music
as its first 1000 rows, so the script does the same.
The three *_backtranslation.csv outputs are then copied into colab_work/dataset/, where
03_extract_backtrans_embeddings.py expects them.
"""
import shutil

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from embeddings_utils import DATASET_DIR, TRANSLATION_DIR

FROM_MODEL = "facebook/wmt19-en-de"
TO_MODEL = "facebook/wmt19-de-en"


def back_translation(tokenizer, model, texts, device='cpu', batch_size=16, max_length=128):
    out = []
    model.to(device)
    for i in range(0, len(texts), batch_size):
        batch_text = [str(x).strip() if str(x).strip() else '.'
                      for x in texts[i:(i + batch_size)]]
        encoders = tokenizer(
            batch_text, return_tensors='pt', padding=True,
            truncation=True, max_length=max_length
        ).to(device)
        with torch.inference_mode():
            output = model.generate(**encoders, max_length=max_length)
        out.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    return out


def back_translate_texts(texts, tokenizer1, model1, tokenizer2, model2, device='cpu',
                         batch_size=16, max_length=128, num_beams=2):
    de = back_translation(tokenizer1, model1, texts=texts, device=device,
                          batch_size=batch_size, max_length=max_length)
    return back_translation(tokenizer2, model2, texts=de, device=device,
                            batch_size=batch_size, max_length=max_length)


def augment_col(series, token_en_de, m_en_de, token_de_en, m_de_en, device='cpu',
                batch_size=32, max_length=128, num_beams=True):
    raw = series.fillna("").astype(str).tolist()
    mapped = []
    for i in tqdm(range(0, len(raw), batch_size)):
        chunk = raw[i:(i + batch_size)]
        mapped.extend(back_translate_texts(
            chunk, token_en_de, m_en_de, token_de_en, m_de_en,
            device=device, batch_size=batch_size, max_length=max_length,
            num_beams=num_beams))
    return pd.Series(mapped, index=series.index, dtype=object)


def load_models():
    model_en_de = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
    model_de_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')
    tokenizer_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
    tokenizer_de = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
    return tokenizer_en, model_en_de, tokenizer_de, model_de_en


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TRANSLATION_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_en, model_en_de, tokenizer_de, model_de_en = load_models()

    def translate(df, source_col, out_name):
        df['Text_backtranslated'] = augment_col(
            df[source_col].astype(str), tokenizer_en, model_en_de,
            tokenizer_de, model_de_en,
            device='cpu', batch_size=32, max_length=128, num_beams=4)
        df.to_csv(TRANSLATION_DIR / out_name)
        return df

    df1 = pd.read_csv(TRANSLATION_DIR / 'df_corpus_trt_aug.csv')
    df1 = translate(df1, 'Text', 'df_corpus_trt_aug_backtranslation.csv')
    df1['L3_rewrite_backtranslated'] = augment_col(
        df1['L3_rewrite'].astype(str), tokenizer_en, model_en_de,
        tokenizer_de, model_de_en,
        device='cpu', batch_size=32, max_length=128, num_beams=4)
    df1.to_csv(TRANSLATION_DIR / 'df_corpus_trt_aug_backtranslation.csv')
    print(df1.columns.tolist())

    magazine = pd.read_csv(TRANSLATION_DIR / 'magazineSubscription_df.csv').iloc[:1000, :]
    translate(magazine, 'text', 'magazineSubscription_df_backtranslation.csv')

    digital = pd.read_csv(TRANSLATION_DIR / 'Digital_Music_df.csv').iloc[:1000, :]
    translate(digital, 'text', 'Digital_Music_df_backtranslation.csv')

    for name in ('df_corpus_trt_aug_backtranslation.csv',
                 'magazineSubscription_df_backtranslation.csv',
                 'Digital_Music_df_backtranslation.csv'):
        shutil.copy(TRANSLATION_DIR / name, DATASET_DIR / name)


if __name__ == '__main__':
    main()
