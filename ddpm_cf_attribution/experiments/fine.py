"""Fine attribution: one row per site and horizon."""

from pathlib import Path

from ddpm_cf_attribution.functions import fine_result, write_json


def main(panel, records, out: Path):
    result = fine_result(panel, records)
    write_json(out, result)
    return result
