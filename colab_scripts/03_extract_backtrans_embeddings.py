"""Embeddings for back-translated and syntax-noised texts, ported from notebook cells 30-35.

Expects 02_back_translation.py to have written the *_backtranslation.csv files into
colab_work/dataset/. Cell 35 in the notebook repeats cell 34 exactly, so it is not repeated here.
"""
from embeddings_utils import DATASET_DIR, MODEL_CONFIGS as ALL_MODEL_CONFIGS, get_embedding_wrapper

MODEL_CONFIGS = {
    k: v for k, v in ALL_MODEL_CONFIGS.items()
    if k in ('MiniLM_L6', 'MongoDB_Leaf', 'Emb_gemma')
}


def csv(name):
    return str(DATASET_DIR / name)


def main():
    get_embedding_wrapper(csv('df_corpus_trt_aug_backtranslation.csv'),
                          text_col='Text_backtranslated', MODEL_CONFIGS=MODEL_CONFIGS, back_trans=True)
    get_embedding_wrapper(csv('df_corpus_trt_aug_backtranslation.csv'),
                          text_col='Text', MODEL_CONFIGS=MODEL_CONFIGS, back_trans=False)
    get_embedding_wrapper(csv('df_corpus_trt_aug_backtranslation.csv'),
                          text_col='Syntax_Noise2', MODEL_CONFIGS=MODEL_CONFIGS, back_trans=False)
    get_embedding_wrapper(csv('Digital_Music_df_backtranslation.csv'),
                          text_col='Text_backtranslated', MODEL_CONFIGS=MODEL_CONFIGS, back_trans=True)
    get_embedding_wrapper(csv('magazineSubscription_df_backtranslation.csv'),
                          text_col='Text_backtranslated', MODEL_CONFIGS=MODEL_CONFIGS, back_trans=True)


if __name__ == '__main__':
    main()
