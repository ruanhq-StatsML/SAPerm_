"""Uplift monitor plus Thompson sampling over the ten sites."""

from ddpm_cf_attribution.uplift_monitor import run_monitor


def main():
    payload = run_monitor()
    pulls = payload["thompson"]["pulls"]
    print(
        f"uplift primary {payload['primary_by_uplift']} "
        f"oracle {payload['oracle_primary']} "
        f"sign agree {payload['n_sign_agree']}/10 "
        f"deployed pulls {pulls} "
        f"regret {payload['thompson']['cumulative_regret']:.3f} "
        f"cold pulls {payload['cold_start']['pulls']} "
        f"alerts {payload['alerts']}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    main()
