"""Per-dataset digestion scripts: read → preprocess → embed/meta → schema."""

from .biasfinder_gen import digest as digest_biasfinder_gen
from .modasense_emotion import digest as digest_modasense

REGISTRY = {
    "biasfinder_gen": digest_biasfinder_gen,
    "modasense_emotion": digest_modasense,
}
