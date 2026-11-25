"""Fit once, run both attributions, merge them into the package outputs."""

from pathlib import Path

from ddpm_cf_attribution.experiments.coarse import main as coarse_main
from ddpm_cf_attribution.experiments.fine import main as fine_main
from ddpm_cf_attribution.functions import merge_results, write_json
from ddpm_cf_attribution.run_experiment import run


def main(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parents[1] / "outputs"
    result, panel, records = run(return_details=True, out_dir=out_dir)
    coarse = coarse_main(panel, records, result["headline"], result["draw_size"], out_dir / "result_coarse.json")
    fine = fine_main(panel, records, out_dir / "result_fine.json")
    merged = merge_results(coarse, fine)
    write_json(out_dir / "result_merged.json", merged)
    print(
        f"merged peak={merged['fine_peak_horizon']} "
        f"sign={merged['sign_matches']} "
        f"post_anomaly_unchanged={merged['post_anomaly_unchanged']}",
        flush=True,
    )
    return merged


if __name__ == "__main__":
    main()
