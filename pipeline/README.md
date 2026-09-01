# pipeline/ — team_19 generation code

| File | Role |
|---|---|
| `priors.py` | The model's elicited beliefs (control baselines, demographic structure, per-intervention × per-outcome effects, partisan moderation, response style). This is the raw model output. |
| `simulate.py` | Seeded simulator: priors → `raw_data_deposit/team_19_raw_export.csv` (Qualtrics variable names/codes). 4,000 profiles × 17 conditions with common random numbers. |
| `clean_py.py` | Python port of `scripts/lib/clean_lib.R` → `predictions/team_19_T1_primary_v1.csv`. |
| `finalize.py` | Python stand-in for `make manifest`, `make zenodo_citation`, `make check`. |

Reproduce (Python ≥3.9, numpy, pandas):

    python3 pipeline/simulate.py
    python3 pipeline/clean_py.py
    python3 pipeline/finalize.py
