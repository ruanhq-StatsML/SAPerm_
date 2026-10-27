"""Coarse attribution: one headline window."""

from pathlib import Path

from ddpm_cf_attribution.functions import coarse_result, write_json


def main(panel, records, headline, draw_size, out: Path):
    result = coarse_result(panel, records, headline, draw_size)
    write_json(out, result)
    return result
