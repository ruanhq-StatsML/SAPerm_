"""FSDS on biasfinder: PermuCATE with the model registry."""

from ddpm_cf_attribution.permucate_aggregate import run_real


def main():
    payload = run_real()
    print(
        f"primary {payload['primary_feature']} "
        f"vimp {payload['primary_vimp']:.4f} "
        f"kept {payload['kept']} dropped {payload['dropped']} "
        f"variance kept {payload['data']['explained_variance_sum']:.3f} "
        f"deployed pulls {payload['thompson']['pulls']} "
        f"most pulled {payload['most_pulled']} "
        f"cold pulls {payload['cold_start']['pulls']} "
        f"cold most {payload['cold_most_pulled']} "
        f"regret {payload['thompson']['cumulative_regret']:.3f}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    main()
