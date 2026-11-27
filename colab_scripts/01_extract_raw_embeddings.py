"""Raw embedding extraction, ported from notebook cells 3-18.

Replaces the Colab steps: the GitHub-token download (cell 5) is replaced by unpacking
datasets/dataset.zip locally, and the Drive-only dependencies are dropped.
Outputs go to colab_work/dataset (*.npy next to each CSV) and colab_work/Embeddings.
"""
import os
import shutil
import zipfile

from embeddings_utils import (
    DATASET_DIR, EMBEDDING_DIR, MODEL_CONFIGS, REPO_ROOT, WORK_DIR, get_embedding_wrapper,
)

ZIP_PATH = REPO_ROOT / 'datasets' / 'dataset.zip'


def prepare_dirs():
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def unpack_dataset():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member in zf.namelist():
            if member.startswith('__MACOSX/') or member.endswith('/'):
                continue
            target = DATASET_DIR / os.path.basename(member)
            with zf.open(member) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)


def main():
    prepare_dirs()
    unpack_dataset()

    def csv(name):
        return str(DATASET_DIR / name)

    get_embedding_wrapper(csv('GEN-sarc-notsarc.csv'), text_col='text', MODEL_CONFIGS=MODEL_CONFIGS)
    get_embedding_wrapper(csv('RQ-sarc-notsarc.csv'), text_col='text', MODEL_CONFIGS=MODEL_CONFIGS)
    get_embedding_wrapper(csv('HYP-sarc-notsarc.csv'), text_col='text', MODEL_CONFIGS=MODEL_CONFIGS)

    small_configs = {k: v for k, v in MODEL_CONFIGS.items() if k in ('MongoDB_Leaf', 'Emb_gemma')}
    get_embedding_wrapper(csv('train_bias2.csv'), text_col=1, MODEL_CONFIGS=small_configs,
                          delimiter='\t', header=None)
    get_embedding_wrapper(csv('test_bias2.csv'), text_col=1, MODEL_CONFIGS=small_configs,
                          delimiter='\t', header=None)
    get_embedding_wrapper(csv('Equity-Evaluation-Corpus.csv'), text_col='Sentence',
                          MODEL_CONFIGS=small_configs)

    archive = shutil.make_archive(str(EMBEDDING_DIR / 'dataset_embedding'), 'zip', str(DATASET_DIR))
    print(f'Wrote {archive} (work dir: {WORK_DIR})')


if __name__ == '__main__':
    main()
