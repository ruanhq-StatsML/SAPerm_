"""Real-data PermuCATE pooled across the ten meta-learners."""

from ddpm_cf_attribution.permucate_aggregate import run_real


def main():
    payload = run_real()
    print(
        f"primary {payload['primary_feature']} "
        f"ivw {payload['primary_ivw']:.4f} "
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
