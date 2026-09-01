#!/usr/bin/env python3
"""
finalize.py — Python stand-ins for `make manifest`, `make zenodo_citation`, `make check`
(scripts/manifest.R, scripts/zenodo_citation.R, scripts/check.R), used because R was
not available on the build machine. Same fingerprint (SHA-256 of file bytes), same
.zenodo.json fields, and the same hard checks (coverage, NA, ranges, floor).

Usage: python3 pipeline/finalize.py
"""
import hashlib
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from clean_py import TIER1_REQUIRED, TRUST_ITEMS  # noqa: E402

INTERVENTIONS = ["Corporate reliance", "Social justice", "Interview Prof. Maraun", "Funding",
                 "Oil industry misinformation", "Measurement & modeling (1)", "Former skeptics",
                 "High public trust", "Measurement & modeling (2)", "Peer-review",
                 "Scientist community helpers", "Consensus", "Portrait Prof. Cherry", "Model accuracy",
                 "Interview Prof. Sebille", "Extreme weather predictions"]
CONDITIONS = ["control"] + INTERVENTIONS
MODERATORS = {
    "gender": ["Male", "Female", "Other"],
    "age_band": ["18-29", "30-44", "45-59", "60+"],
    "race": ["White / Caucasian", "Black / African American", "Hispanic / Latino",
             "Asian / Asian American", "Other"],
    "education": ["Less than high school", "High school diploma / GED",
                  "Some college or Associate's degree", "Bachelor's degree",
                  "Master's degree / Professional degree", "Doctorate degree / Ph.D."],
    "income": ["Less than $30,000", "$30,000 to $55,999", "$56,000 to $99,999",
               "$100,000 to $167,999", "$168,000 or more"],
    "party": ["Republican", "Democrat", "Independent", "Other"],
}
SCALE_0_100 = ["trust_multidimensional", "trust_post", "distrust_post", "funding_perceptions",
               "policy_role_mean", "inst_trust_mean", "belief_post", "concern_mean", "policy_general",
               "policy_specific_mean", "behavior_mean"] + TRUST_ITEMS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(m):
    team, tier, entry = m["team_id"], int(m["tier"]), m["entry"]
    pat = re.compile(rf"^{re.escape(team)}_T{tier}_{re.escape(entry)}_v\d+\.csv$")
    files = sorted(f for f in os.listdir(os.path.join(ROOT, "predictions")) if pat.match(f))
    assert files, "no prediction files matching the naming grammar"
    m["prediction_files"] = [{"file": f"predictions/{f}", "sha256": sha256(os.path.join(ROOT, "predictions", f))}
                             for f in files]
    return m


def zenodo(m):
    desc = {1: "individual simulation", 2: "group-level reasoning", 3: "direct effect forecast"}[int(m["tier"])]
    creators = []
    for c in m["creators"]:
        out = {"name": c["name"]}
        if c.get("affiliation"):
            out["affiliation"] = c["affiliation"]
        if c.get("orcid"):
            out["orcid"] = c["orcid"]   # only a valid ORCID may be placed here (checked in check())
        creators.append(out)
    paras = [p.strip() for p in re.split(r"\n[ \t]*\n", m["abstract"].strip()) if p.strip()]
    z = {
        "upload_type": "software",
        "title": f"Silicon Sample Benchmark — Tier {m['tier']} ({desc}) submission (team {m['team_id']})",
        "version": f"1.0-t{m['tier']}",
        "language": "eng",
        "access_right": "open",
        "license": m.get("license", "CC-BY-4.0"),
        "creators": creators,
        "description": "".join(f"<p>{p}</p>" for p in paras),
        "keywords": ["Silicon Sample Benchmark", "silicon sampling", "large language models",
                     "computational social science", "survey methodology", "public opinion",
                     "climate communication", "treatment effect prediction"],
        "related_identifiers": [
            {"identifier": "https://janpfander.github.io/llm_predictions_megastudy/",
             "relation": "isPartOf", "scheme": "url", "resource_type": "publication-other"},
            {"identifier": "https://janpfander.github.io/llm_predictions_megastudy/amendment_preregistration.html",
             "relation": "references", "scheme": "url", "resource_type": "publication-preprint"},
            {"identifier": m["code_repository"], "relation": "isCompiledBy", "scheme": "url",
             "resource_type": "software"},
        ],
        "notes": f"Team {m['team_id']}. Disclosure class {m.get('disclosure_class', 'A')}.",
    }
    return z


def orcid_ok(x):
    d = x.replace("-", "")
    if not re.fullmatch(r"[0-9]{15}[0-9X]", d):
        return False
    total = 0
    for ch in d[:15]:
        total = (total + int(ch)) * 2
    r = (12 - total % 11) % 11
    return ("X" if r == 10 else str(r)) == d[15]


def check(m):
    fails, warns = [], []
    assert int(m["tier"]) == 1
    assert m["coverage"] == {"interventions": 16, "outcomes": 13}
    assert m["blinding_attestation"] is True
    for c in m["creators"]:
        if c.get("orcid") and not orcid_ok(c["orcid"]):
            fails.append(f"invalid ORCID {c['orcid']}")
    for pf in m["prediction_files"]:
        path = os.path.join(ROOT, pf["file"])
        if sha256(path) != pf["sha256"]:
            fails.append(f"fingerprint mismatch {pf['file']}")
        d = pd.read_csv(path)
        missing = [c for c in TIER1_REQUIRED if c not in d.columns]
        if missing:
            fails.append(f"missing columns {missing}")
        if d[TIER1_REQUIRED].isna().sum().sum():
            fails.append("NA values present")
        if d["profile_id"].duplicated().any():
            fails.append("duplicate profile_id")
        conds = set(d["condition"])
        if conds != set(CONDITIONS):
            fails.append(f"condition set mismatch: {conds ^ set(CONDITIONS)}")
        n = d["condition"].value_counts()
        low = [c for c in INTERVENTIONS if n.get(c, 0) < 500] + (["control"] if n.get("control", 0) < 1000 else [])
        if low:
            warns.append(f"below precision floor: {low}")
        for mod, levels in MODERATORS.items():
            bad = set(d[mod]) - set(levels)
            if bad:
                fails.append(f"{mod} has invalid levels {bad}")
        for col in SCALE_0_100:
            if ((d[col] < 0) | (d[col] > 100)).any():
                fails.append(f"{col} out of [0,100]")
        if ((d["donation_ams"] < 0) | (d["donation_ams"] > 10)).any():
            fails.append("donation_ams out of [0,10]")
        if not set(d["newsletter_signup"].unique()) <= {0, 1}:
            fails.append("newsletter_signup not binary")
        if (d["trust_multidimensional"] - d[TRUST_ITEMS].mean(axis=1)).abs().max() > 1e-6:
            warns.append("trust_multidimensional inconsistent with items")
        print(f"[ok] {pf['file']}: {len(d)} rows, {len(conds)} conditions, min N/condition {n.min()}")
    for f in os.listdir(os.path.join(ROOT, "predictions")):
        if f.startswith("example_"):
            fails.append(f"example file still present: {f}")
    if os.path.exists(os.path.join(ROOT, "raw_data_deposit", "example_raw_export.csv")):
        fails.append("example raw export still present")
    reg = open(os.path.join(ROOT, "registration.md")).read()
    blank = [ln for ln in reg.splitlines() if ln.startswith("- **") and ln.rstrip().endswith(":")]
    if blank:
        warns.append(f"{len(blank)} registration.md items still blank")
    return fails, warns


if __name__ == "__main__":
    mp = os.path.join(ROOT, "metadata.json")
    m = json.load(open(mp))
    m = manifest(m)
    json.dump(m, open(mp, "w"), indent=2, ensure_ascii=False)
    open(mp, "a").write("\n")
    for pf in m["prediction_files"]:
        print("manifest:", pf["file"], pf["sha256"])
    z = zenodo(m)
    json.dump(z, open(os.path.join(ROOT, ".zenodo.json"), "w"), indent=2, ensure_ascii=False)
    open(os.path.join(ROOT, ".zenodo.json"), "a").write("\n")
    print("zenodo: wrote .zenodo.json")
    fails, warns = check(m)
    for w in warns:
        print("[warn]", w)
    for f in fails:
        print("[FAIL]", f)
    print("OVERALL:", "FAIL" if fails else ("PASS-WITH-WARNINGS" if warns else "PASS"))
    sys.exit(1 if fails else 0)
