#!/usr/bin/env python3
"""
simulate.py — deterministic simulator: priors.py -> raw Qualtrics-format export.

Design: N_PROFILES synthetic respondents are drawn once (demographics, latent
traits, item-level noise, response-style draws). Each profile is then run through
ALL 17 conditions with those same random numbers ("common random numbers"), so
the file contains exact counterfactuals and between-condition sampling noise in
baselines is eliminated. Rows are independent respondents from the scorer's point
of view (profile_id is unique per row: <profile>_c<condition index>).

Writes raw_data_deposit/team_19_raw_export.csv with Qualtrics variable names and
numeric demographic codes, as a Qualtrics export of the shipped survey would look
for the scored columns. scripts/clean.R (or pipeline/clean_py.py) then builds the
analysis-ready Tier-1 file. Nothing here touches any human data.

Usage: python3 pipeline/simulate.py [--out PATH]
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import priors as P  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDITIONS = ["control"] + list(P.INTERVENTIONS.keys())
ITEM_NAMES = list(P.ITEMS.keys())
CODE = {
    "gender": {"Male": 1, "Female": 2, "Other": 3},
    "race": {r: i + 1 for i, r in enumerate(P.RACES)},
    "education": {e: i + 1 for i, e in enumerate(P.EDUCATION)},
    "income": {e: i + 1 for i, e in enumerate(P.INCOME)},
    "party": {"Republican": 1, "Democrat": 2, "Independent": 3, "Other": 4},
}


def edu_band(edu):
    if edu in ("Less than high school", "High school diploma / GED"):
        return "low"
    if edu == "Some college or Associate's degree":
        return "mid"
    return "high"


def softmax_adjust(p, adj):
    logit = np.log(np.asarray(p, dtype=float)) + np.asarray(adj, dtype=float)
    e = np.exp(logit - logit.max())
    return e / e.sum()


def draw_profile(rng):
    age_band = rng.choice(P.AGE_BANDS, p=P.AGE_P)
    if rng.random() < P.P_OTHER_GENDER:
        gender = "Other"
        race_w = np.array(P.RACE_GIVEN_GENDER["Male"]) + np.array(P.RACE_GIVEN_GENDER["Female"])
    else:
        gender = "Male" if rng.random() < P.MALE_GIVEN_AGE[age_band] else "Female"
        race_w = np.array(P.RACE_GIVEN_GENDER[gender], dtype=float)
    race = rng.choice(P.RACES, p=race_w / race_w.sum())
    edu = rng.choice(P.EDUCATION, p=P.EDU_P_YOUNG if age_band == "18-29" else P.EDU_P)
    eb = edu_band(edu)
    income = rng.choice(P.INCOME, p=P.INCOME_GIVEN_EDU[eb])
    adj = np.zeros(4)
    if gender == "Female":
        adj += P.PARTY_ADJ_FEMALE
    adj += P.PARTY_ADJ_AGE[age_band]
    if eb == "high":
        adj += P.PARTY_ADJ_EDU_HIGH
    elif eb == "low":
        adj += P.PARTY_ADJ_EDU_LOW
    party = rng.choice(P.PARTIES, p=softmax_adjust(P.PARTY_GIVEN_RACE[race], adj))
    lo, hi = {"18-29": (18, 29), "30-44": (30, 44), "45-59": (45, 59), "60+": (60, 88)}[age_band]
    age = int(min(88, 60 + rng.exponential(8.0))) if age_band == "60+" else int(rng.integers(lo, hi + 1))
    return dict(gender=gender, age_band=age_band, race=race, education=edu, edu_band=eb,
                income=income, party=party, year_birth=2026 - age,
                control_filler=rng.choice(["neckties", "baseball", "dances"]))


def draw_latents(rng, prof):
    mu = P.THETA_PARTY[prof["party"]]
    if prof["gender"] == "Female":
        mu += P.THETA_ADJ["female"]
    if prof["age_band"] in ("18-29", "60+"):
        mu += P.THETA_ADJ[prof["age_band"]]
    if prof["edu_band"] == "high":
        mu += P.THETA_ADJ["edu_high"]
    elif prof["edu_band"] == "low":
        mu += P.THETA_ADJ["edu_low"]
    mu += P.THETA_ADJ.get(prof["race"], 0.0)
    theta = mu + rng.normal(0, P.THETA_RESID_SD)
    tau = P.TAU_THETA_LOAD * theta + rng.normal(0, math.sqrt(1 - P.TAU_THETA_LOAD ** 2))
    beta = P.BETA_THETA_LOAD * theta + rng.normal(0, math.sqrt(1 - P.BETA_THETA_LOAD ** 2))
    return theta, tau, beta


def draw_noise(rng):
    """All per-profile random numbers reused across conditions (common random numbers)."""
    return dict(
        sub={s: rng.normal(0, sd) for s, sd in P.SUBSCALE_SD.items()},
        item={it: rng.normal(0, P.ITEMS[it][3]) for it in ITEM_NAMES},
        heap_u={it: rng.random() for it in ITEM_NAMES},
        snap_u={it: rng.random() for it in ITEM_NAMES},
        effect_z=rng.normal(),
        don_e=rng.normal(0, P.DONATION["noise_sd"]),
        news_u=rng.random(),
    )


def heap(x, u, snap_u):
    """Integer slider response with human-like heaping (deterministic given u, snap_u)."""
    x = float(np.clip(x, 0, 100))
    H = P.HEAP
    if u < H["p_mid50"] and abs(x - 50) < 12:
        return 50
    if u < H["p_mid50"] + H["p_round10"]:
        v = int(round(x / 10.0) * 10)
    elif u < H["p_mid50"] + H["p_round10"] + H["p_round5"]:
        v = int(round(x / 5.0) * 5)
    else:
        v = int(round(x))
    s = H["endpoint_snap"]
    if v >= 100 - s and snap_u < H["p_snap"]:
        v = 100
    elif v <= s and snap_u < H["p_snap"]:
        v = 0
    return int(np.clip(v, 0, 100))


def effects_for(prof, theta, cond, effect_z):
    """Per-respondent effect on the trust composite (points) and per-outcome effects."""
    if cond == "control":
        return 0.0, {}
    spec = P.INTERVENTIONS[cond]
    mult = spec["party"][prof["party"]]
    mult *= P.EFFECT_EDU[prof["edu_band"]] * P.EFFECT_AGE[prof["age_band"]]
    mult *= float(np.clip(1 + P.EFFECT_THETA_SLOPE * theta, 0.5, 1.5))
    mult *= float(np.clip(1 + P.EFFECT_HETERO_SD * effect_z, 0.0, 2.0))
    d = spec["base"] * mult
    w = spec["sub"]
    wsum = sum(w[s] for s in P.SUBSCALES) / 4.0
    eff = {s: d * w[s] / wsum for s in P.SUBSCALES}
    for key, frac in P.SPILL.items():
        eff[key] = (spec["override"][key] if key in spec["override"] else spec["base"] * frac) * mult
    for key in spec["override"]:
        if key not in eff:
            eff[key] = spec["override"][key] * mult
    return d, eff


def respond(prof, theta, tau, beta, nz, cond):
    d, eff = effects_for(prof, theta, cond, nz["effect_z"])
    row = {}
    for item, (mean, gap, tload, nsd, sub) in P.ITEMS.items():
        y = mean + (gap / 1.5) * theta + tload * tau + nz["sub"][sub] + nz["item"][item]
        if sub == "behav":
            y += P.BEHAV_BETA_LOAD * beta
        if cond != "control":
            if sub in P.SUBSCALES:
                y += eff[sub]
            elif sub == "funding":
                y -= eff["funding"]  # raw funding_5 is reverse-scored
            elif sub == "inst":
                y += eff["inst"] * (P.INST_FEDGOV_FRAC if item == "inst_trust_gov_1" else 1.0)
            elif sub in eff:
                y += eff[sub]
        row[item] = heap(y, nz["heap_u"][item], nz["snap_u"][item])
    D = P.DONATION
    lat = D["mean_latent"] + D["theta_load"] * theta + D["beta_load"] * beta + nz["don_e"]
    lat += P.DONATION_PER_POINT * d
    row["donation"] = 0 if lat < D["zero_below"] else int(np.clip(round(lat), 0, 10))
    Nl = P.NEWSLETTER
    logit = math.log(Nl["base_p"] / (1 - Nl["base_p"])) + Nl["theta_load"] * theta + Nl["beta_load"] * beta
    logit += P.NEWSLETTER_LOGIT_PER_POINT * d
    row["newsletter"] = 1 if nz["news_u"] < 1 / (1 + math.exp(-logit)) else 2
    return row


def simulate(out_path):
    rng = np.random.default_rng(P.SEED)
    fieldnames = (["profile_id", "profile", "condition", "control_filler", "gender", "year_birth",
                   "race", "education", "income", "party"] + ITEM_NAMES + ["donation", "newsletter"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_rows = 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(1, P.N_PROFILES + 1):
            prof = draw_profile(rng)
            theta, tau, beta = draw_latents(rng, prof)
            nz = draw_noise(rng)
            base = {
                "profile": f"t19_{i:05d}",
                "gender": CODE["gender"][prof["gender"]],
                "year_birth": prof["year_birth"],
                "race": CODE["race"][prof["race"]],
                "education": CODE["education"][prof["education"]],
                "income": CODE["income"][prof["income"]],
                "party": CODE["party"][prof["party"]],
            }
            for ci, cond in enumerate(CONDITIONS):
                row = dict(base)
                row["profile_id"] = f"t19_{i:05d}_c{ci:02d}"
                row["condition"] = cond
                row["control_filler"] = prof["control_filler"] if cond == "control" else ""
                row.update(respond(prof, theta, tau, beta, nz, cond))
                w.writerow(row)
                n_rows += 1
    print(f"wrote {n_rows} rows ({P.N_PROFILES} profiles x {len(CONDITIONS)} conditions) to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "raw_data_deposit", "team_19_raw_export.csv"))
    simulate(ap.parse_args().out)
