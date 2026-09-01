#!/usr/bin/env python3
"""
clean_py.py — Python port of scripts/lib/clean_lib.R (same recodes and composites).

Used because R was not available on the build machine at deposit time. It
reproduces clean_submission() exactly: Qualtrics label -> target label rename,
demographic code -> label, funding_perceptions = 100 - funding_5, newsletter
Yes/No -> 1/0, age_band from year_birth, and the mean composites. Column order
follows submission_spec.R tier1_required.
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RENAME = {
    "trust_competent_1": "trust_competence_1", "trust_intelligent_1": "trust_competence_2",
    "trust_qualified_1": "trust_competence_3", "trust_honest_1": "trust_integrity_1",
    "trust_ethical_1": "trust_integrity_2", "trust_sincere_1": "trust_integrity_3",
    "trust_concerned_1": "trust_benevolence_1", "trust_improve_1": "trust_benevolence_2",
    "trust_considerate_1": "trust_benevolence_3", "trust_feedback_1": "trust_openness_1",
    "trust_transparent_1": "trust_openness_2", "trust_attention_1": "trust_openness_3",
    "trust_post_1": "trust_post", "distrust_1": "distrust_post", "donation": "donation_ams",
    "newsletter": "newsletter_signup", "funding_5": "funding_perceptions",
    "policy_1_1": "policy_role_1", "policy_2_1": "policy_role_2", "policy_3_1": "policy_role_3",
    "policy_4_1": "policy_role_4", "inst_trust_epa_1": "inst_trust_epa",
    "inst_trust_nasa_1": "inst_trust_nasa", "inst_trust_noaa_1": "inst_trust_noaa",
    "inst_trust_uni_1": "inst_trust_universities", "inst_trust_gov_1": "inst_trust_federal_gov",
    "belief_post_1": "belief_post", "concern_1_1": "concern_1", "concern_2_1": "concern_2",
    "concern_3_1": "concern_3", "policy_general_1": "policy_general",
    **{f"policy_specific_{i}_1": f"policy_specific_{i}" for i in range(1, 8)},
    "individual_meat_1": "behavior_meat", "individual_transport_1": "behavior_transport",
    "individual_solar_1": "behavior_solar", "individual_fly_1": "behavior_fly",
    "individual_talk_1": "behavior_talk", "individual_donate_1": "behavior_donate",
}
GENDER = {1: "Male", 2: "Female", 3: "Other"}
RACE = {1: "White / Caucasian", 2: "Black / African American", 3: "Hispanic / Latino",
        4: "Asian / Asian American", 5: "Other"}
EDU = {1: "Less than high school", 2: "High school diploma / GED",
       3: "Some college or Associate's degree", 4: "Bachelor's degree",
       5: "Master's degree / Professional degree", 6: "Doctorate degree / Ph.D."}
INCOME = {1: "Less than $30,000", 2: "$30,000 to $55,999", 3: "$56,000 to $99,999",
          4: "$100,000 to $167,999", 5: "$168,000 or more"}
PARTY = {1: "Republican", 2: "Democrat", 3: "Independent", 4: "Other"}
TRUST_ITEMS = [f"trust_{s}_{i}" for s in ("competence", "integrity", "benevolence", "openness") for i in (1, 2, 3)]
TIER1_REQUIRED = (["profile_id", "condition", "gender", "age_band", "race", "education", "income", "party",
                   "trust_multidimensional"] + TRUST_ITEMS +
                  ["trust_post", "distrust_post", "funding_perceptions", "policy_role_mean", "inst_trust_mean",
                   "belief_post", "concern_mean", "policy_general", "policy_specific_mean", "behavior_mean",
                   "donation_ams", "newsletter_signup"])


def age_band(age):
    if age <= 29:
        return "18-29"
    if age <= 44:
        return "30-44"
    if age <= 59:
        return "45-59"
    return "60+"


def clean(raw_path, out_path):
    d = pd.read_csv(raw_path).rename(columns=RENAME)
    d["gender"] = d["gender"].map(GENDER)
    d["race"] = d["race"].map(RACE)
    d["education"] = d["education"].map(EDU)
    d["income"] = d["income"].map(INCOME)
    d["party"] = d["party"].map(PARTY)
    d["age_band"] = (2026 - d["year_birth"]).map(age_band)
    d["funding_perceptions"] = 100 - d["funding_perceptions"]
    d["newsletter_signup"] = d["newsletter_signup"].map({1: 1, 2: 0})
    for s in ("competence", "integrity", "benevolence", "openness"):
        d[f"trust_{s}"] = d[[f"trust_{s}_{i}" for i in (1, 2, 3)]].mean(axis=1)
    d["trust_multidimensional"] = d[["trust_competence", "trust_integrity", "trust_benevolence", "trust_openness"]].mean(axis=1)
    d["policy_role_mean"] = d[[f"policy_role_{i}" for i in range(1, 5)]].mean(axis=1)
    d["inst_trust_mean"] = d[["inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa",
                              "inst_trust_universities", "inst_trust_federal_gov"]].mean(axis=1)
    d["concern_mean"] = d[["concern_1", "concern_2", "concern_3"]].mean(axis=1)
    d["policy_specific_mean"] = d[[f"policy_specific_{i}" for i in range(1, 8)]].mean(axis=1)
    d["behavior_mean"] = d[["behavior_meat", "behavior_transport", "behavior_solar",
                            "behavior_fly", "behavior_talk", "behavior_donate"]].mean(axis=1)
    out = d[TIER1_REQUIRED]
    assert out.isna().sum().sum() == 0, "NA values in cleaned output"
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path}")
    return out


if __name__ == "__main__":
    m = json.load(open(os.path.join(ROOT, "metadata.json")))
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "raw_data_deposit", "team_19_raw_export.csv")
    out = os.path.join(ROOT, "predictions", f"{m['team_id']}_T1_{m['entry']}_v1.csv")
    clean(raw, out)
