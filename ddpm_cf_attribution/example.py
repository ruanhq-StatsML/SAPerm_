"""One call. Writes the report under ddpm_cf_attribution/outputs/."""

import ddpm_cf_attribution

result = ddpm_cf_attribution.run()
print(result["primary_driver"]["estimated_site"], result["headline"]["delta"])
