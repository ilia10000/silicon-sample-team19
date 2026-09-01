"""
priors.py — the model's elicited beliefs, written out as explicit parameters.

This file IS the "raw model output" of this submission. Every number here was
written by the language model (Claude, via Claude Code) after reading the full
survey instrument (survey/questionnaire.txt), the codebook, and all sixteen
intervention stimulus texts, in a single interactive session on 2026-08-31.
No numbers were supplied by the human team member, and nothing here was fit
to any data from this study. simulate.py turns these parameters into synthetic
respondents deterministically (seeded).

Units: all slider quantities are points on the 0-100 scale. "gap" is the
expected Democrat-minus-Republican difference in the control condition.
"""

# ----------------------------------------------------------------------------
# 1. Population structure (control-condition baselines)
# ----------------------------------------------------------------------------

# Census cross-quotas from the benchmark preregistration (gender x age, gender x race)
AGE_BANDS = ["18-29", "30-44", "45-59", "60+"]
AGE_P = [0.202, 0.260, 0.229, 0.309]
MALE_GIVEN_AGE = {"18-29": 0.509, "30-44": 0.505, "45-59": 0.497, "60+": 0.461}
P_OTHER_GENDER = 0.008  # not in the quota table; small share in opt-in panels

RACES = ["White / Caucasian", "Black / African American", "Hispanic / Latino",
         "Asian / Asian American", "Other"]
RACE_GIVEN_GENDER = {  # from the quota table's gender x race counts
    "Male":   [5332, 1042, 1646, 568, 240],
    "Female": [5500, 1170, 1617, 633, 252],
}

EDUCATION = ["Less than high school", "High school diploma / GED",
             "Some college or Associate's degree", "Bachelor's degree",
             "Master's degree / Professional degree", "Doctorate degree / Ph.D."]
EDU_P = [0.07, 0.27, 0.29, 0.23, 0.11, 0.03]
EDU_P_YOUNG = [0.06, 0.30, 0.34, 0.22, 0.07, 0.01]   # 18-29 (fewer graduate degrees yet)

INCOME = ["Less than $30,000", "$30,000 to $55,999", "$56,000 to $99,999",
          "$100,000 to $167,999", "$168,000 or more"]
# income distribution conditional on education band (low / mid / high)
INCOME_GIVEN_EDU = {
    "low":  [0.36, 0.30, 0.22, 0.09, 0.03],   # <HS, HS
    "mid":  [0.22, 0.25, 0.30, 0.17, 0.06],   # some college
    "high": [0.10, 0.15, 0.30, 0.29, 0.16],   # BA+
}

PARTIES = ["Republican", "Democrat", "Independent", "Other"]
# base party distribution by race (opt-in panel; Independents run high)
PARTY_GIVEN_RACE = {
    "White / Caucasian":        [0.41, 0.21, 0.33, 0.05],
    "Black / African American": [0.08, 0.58, 0.30, 0.04],
    "Hispanic / Latino":        [0.24, 0.35, 0.36, 0.05],
    "Asian / Asian American":   [0.20, 0.40, 0.35, 0.05],
    "Other":                    [0.25, 0.30, 0.38, 0.07],
}
# additive log-odds adjustments (R, D, I, Other)
PARTY_ADJ_FEMALE = [-0.15, 0.15, 0.0, 0.0]
PARTY_ADJ_AGE = {"18-29": [-0.25, 0.0, 0.15, 0.1], "30-44": [-0.05, 0.0, 0.05, 0.0],
                 "45-59": [0.05, 0.0, 0.0, 0.0], "60+": [0.2, 0.05, -0.15, -0.1]}
PARTY_ADJ_EDU_HIGH = [-0.12, 0.15, 0.0, 0.0]   # BA+
PARTY_ADJ_EDU_LOW = [0.1, -0.05, 0.0, 0.0]     # <HS, HS

# Latent "climate orientation" theta (SD units): mean by party, plus adjustments.
THETA_PARTY = {"Republican": -0.75, "Democrat": 0.75, "Independent": -0.05, "Other": -0.15}
THETA_ADJ = {
    "female": 0.10, "18-29": 0.15, "60+": -0.10, "edu_high": 0.10, "edu_low": -0.05,
    "Black / African American": 0.05, "Hispanic / Latino": 0.10, "Asian / Asian American": 0.15,
}
THETA_RESID_SD = 0.75
# Second latent tau: generalized institutional trust / deference (corr ~0.4 with theta)
TAU_THETA_LOAD = 0.4
# Third latent beta: action propensity (donating, subscribing, behaviors), corr ~0.5 with theta
BETA_THETA_LOAD = 0.5

# ----------------------------------------------------------------------------
# 2. Item-level control baselines: (overall mean, D-R gap, tau loading, item noise SD)
#    slope on theta is gap / 1.5 (the two party means of theta differ by 1.5)
# ----------------------------------------------------------------------------
ITEMS = {
    # trust items (Qualtrics label): mean, gap, tau_load, noise_sd, subscale
    "trust_competent_1":   (68, 22, 8, 11, "competence"),
    "trust_intelligent_1": (71, 20, 8, 11, "competence"),
    "trust_qualified_1":   (69, 22, 8, 11, "competence"),
    "trust_honest_1":      (62, 30, 9, 11, "integrity"),
    "trust_ethical_1":     (63, 29, 9, 11, "integrity"),
    "trust_sincere_1":     (63, 29, 9, 11, "integrity"),
    "trust_concerned_1":   (65, 26, 8, 12, "benevolence"),
    "trust_improve_1":     (62, 25, 8, 12, "benevolence"),
    "trust_considerate_1": (60, 26, 8, 12, "benevolence"),
    "trust_feedback_1":    (58, 26, 8, 12, "openness"),
    "trust_transparent_1": (58, 27, 8, 12, "openness"),
    "trust_attention_1":   (55, 25, 8, 12, "openness"),
    # single items
    "trust_post_1":        (62, 34, 9, 12, "trust1"),
    "distrust_1":          (32, -30, -9, 14, "distrust"),
    "funding_5":           (44, -30, -4, 16, "funding"),   # raw: 0 too little ... 100 too much
    # scientists' role in policy
    "policy_1_1": (68, 22, 5, 14, "role"),
    "policy_2_1": (55, 26, 5, 16, "role"),
    "policy_3_1": (74, 16, 5, 13, "role"),
    "policy_4_1": (62, 26, 5, 15, "role"),
    # institutional trust
    "inst_trust_epa_1":  (55, 22, 12, 12, "inst"),
    "inst_trust_nasa_1": (70, 10, 12, 12, "inst"),
    "inst_trust_noaa_1": (64, 14, 12, 12, "inst"),
    "inst_trust_uni_1":  (58, 26, 12, 12, "inst"),
    "inst_trust_gov_1":  (38, -8, 14, 14, "inst"),
    # belief / concern
    "belief_post_1": (66, 36, 4, 14, "belief"),
    "concern_1_1":   (57, 40, 3, 13, "concern"),
    "concern_2_1":   (60, 40, 3, 13, "concern"),
    "concern_3_1":   (46, 38, 3, 15, "concern"),
    # policy
    "policy_general_1":    (63, 42, 4, 14, "polgen"),
    "policy_specific_1_1": (45, 40, 3, 15, "polspec"),
    "policy_specific_2_1": (63, 26, 3, 15, "polspec"),
    "policy_specific_3_1": (72, 26, 3, 14, "polspec"),
    "policy_specific_4_1": (80, 12, 3, 13, "polspec"),
    "policy_specific_5_1": (30, 26, 3, 16, "polspec"),
    "policy_specific_6_1": (66, 32, 3, 14, "polspec"),
    "policy_specific_7_1": (84, 8, 3, 12, "polspec"),
    # behaviors (also load on beta)
    "individual_meat_1":      (38, 22, 0, 16, "behav"),
    "individual_transport_1": (36, 18, 0, 16, "behav"),
    "individual_solar_1":     (27, 12, 0, 16, "behav"),
    "individual_fly_1":       (36, 14, 0, 16, "behav"),
    "individual_talk_1":      (40, 30, 0, 16, "behav"),
    "individual_donate_1":    (26, 22, 0, 15, "behav"),
}
# shared-subscale component SD (items in the same subscale co-move beyond theta/tau)
SUBSCALE_SD = {"competence": 8, "integrity": 8, "benevolence": 8, "openness": 8,
               "trust1": 0, "distrust": 0, "funding": 0, "role": 9, "inst": 8,
               "belief": 0, "concern": 12, "polgen": 0, "polspec": 9, "behav": 10}
BEHAV_BETA_LOAD = 9  # behaviors additionally load on the action-propensity latent

# donation ($0-10 of a $10 bonus) and newsletter signup
DONATION = {"mean_latent": 2.4, "theta_load": 1.1, "beta_load": 1.6, "noise_sd": 2.6,
            "zero_below": 1.6}   # latent < zero_below -> $0 (zero inflation)
NEWSLETTER = {"base_p": 0.035, "theta_load": 0.55, "beta_load": 0.6}

# slider response style: heaping and endpoint snapping
HEAP = {"p_round5": 0.25, "p_round10": 0.12, "p_mid50": 0.04, "endpoint_snap": 3, "p_snap": 0.5}

# ----------------------------------------------------------------------------
# 3. Treatment effects. Base effect = points on trust_multidimensional (the primary),
#    for an average respondent. Subscale weights redistribute it; party multipliers
#    moderate it; spillover fractions carry it to other outcomes unless overridden.
# ----------------------------------------------------------------------------
SUBSCALES = ["competence", "integrity", "benevolence", "openness"]

INTERVENTIONS = {
    "Oil industry misinformation": dict(
        base=3.0, sub=dict(competence=0.8, integrity=1.3, benevolence=0.9, openness=0.9),
        party=dict(Republican=0.6, Democrat=1.2, Independent=1.0, Other=0.9),
        override=dict(belief=2.0, distrust=-2.8, polgen=1.5, funding=1.0, concern=1.0)),
    "Peer-review": dict(
        base=2.5, sub=dict(competence=1.1, integrity=1.1, benevolence=0.6, openness=1.1),
        party=dict(Republican=1.1, Democrat=0.9, Independent=1.0, Other=1.0), override={}),
    "Interview Prof. Maraun": dict(
        base=2.5, sub=dict(competence=0.9, integrity=1.1, benevolence=0.6, openness=1.4),
        party=dict(Republican=1.1, Democrat=0.9, Independent=1.0, Other=1.0), override={}),
    "Consensus": dict(
        base=2.2, sub=dict(competence=1.1, integrity=1.0, benevolence=0.8, openness=0.9),
        party=dict(Republican=1.1, Democrat=0.9, Independent=1.0, Other=1.0),
        override=dict(belief=3.0, concern=1.0, polgen=1.2)),
    "Scientist community helpers": dict(
        base=2.5, sub=dict(competence=0.6, integrity=0.9, benevolence=1.5, openness=0.9),
        party=dict(Republican=1.2, Democrat=0.8, Independent=1.0, Other=1.0),
        override=dict(concern=0.8)),
    "Extreme weather predictions": dict(
        base=2.5, sub=dict(competence=1.2, integrity=0.7, benevolence=1.3, openness=0.7),
        party=dict(Republican=1.2, Democrat=0.8, Independent=1.0, Other=1.0),
        override=dict(inst=1.2, concern=1.2)),
    "Corporate reliance": dict(
        base=2.0, sub=dict(competence=1.4, integrity=1.0, benevolence=0.5, openness=0.6),
        party=dict(Republican=1.3, Democrat=0.8, Independent=1.0, Other=1.0), override={}),
    "Former skeptics": dict(
        base=2.0, sub=dict(competence=0.9, integrity=1.1, benevolence=0.8, openness=1.0),
        party=dict(Republican=1.5, Democrat=0.6, Independent=1.0, Other=1.0),
        override=dict(belief=1.5)),
    "Model accuracy": dict(
        base=2.0, sub=dict(competence=1.5, integrity=0.9, benevolence=0.6, openness=0.6),
        party=dict(Republican=1.1, Democrat=0.9, Independent=1.0, Other=1.0),
        override=dict(belief=1.8)),
    "Funding": dict(
        base=2.0, sub=dict(competence=0.6, integrity=1.4, benevolence=0.9, openness=0.9),
        party=dict(Republican=1.1, Democrat=0.9, Independent=1.0, Other=1.0),
        override=dict(funding=3.0, role=-1.0)),
    "Measurement & modeling (1)": dict(
        base=1.5, sub=dict(competence=1.5, integrity=0.6, benevolence=0.6, openness=0.6),
        party=dict(Republican=1.0, Democrat=1.0, Independent=1.0, Other=1.0),
        override=dict(belief=0.8)),
    "Measurement & modeling (2)": dict(
        base=1.0, sub=dict(competence=1.4, integrity=0.5, benevolence=0.5, openness=0.5),
        party=dict(Republican=1.0, Democrat=1.0, Independent=1.0, Other=1.0), override={}),
    "Portrait Prof. Cherry": dict(
        base=1.5, sub=dict(competence=0.7, integrity=0.8, benevolence=1.3, openness=0.9),
        party=dict(Republican=1.2, Democrat=0.9, Independent=1.0, Other=1.0), override={}),
    "Interview Prof. Sebille": dict(
        base=1.5, sub=dict(competence=0.5, integrity=0.9, benevolence=1.5, openness=0.8),
        party=dict(Republican=0.8, Democrat=1.1, Independent=1.0, Other=1.0),
        override=dict(concern=1.2)),
    "High public trust": dict(
        base=1.5, sub=dict(competence=1.0, integrity=1.0, benevolence=1.0, openness=1.0),
        party=dict(Republican=1.2, Democrat=0.9, Independent=1.0, Other=1.0),
        override=dict(inst=0.5)),
    "Social justice": dict(
        base=0.6, sub=dict(competence=0.4, integrity=1.0, benevolence=1.2, openness=0.6),
        party=dict(Republican=-0.8, Democrat=1.5, Independent=0.6, Other=0.8),
        override=dict(polgen=1.0, concern=0.5)),
}

# spillover of the (party-moderated) trust effect to other outcomes, as a fraction
SPILL = {
    "trust1": 1.15, "distrust": -0.9, "funding": 0.35, "role": 0.3, "inst": 0.25,
    "belief": 0.35, "concern": 0.3, "polgen": 0.35, "polspec": 0.2, "behav": 0.15,
}
INST_FEDGOV_FRAC = 0.4          # federal-gov item gets only this fraction of the inst spillover
DONATION_PER_POINT = 0.05       # dollars per trust point
NEWSLETTER_LOGIT_PER_POINT = 0.05  # logit units per trust point
# additional individual-level moderation of the effect
EFFECT_HETERO_SD = 0.3          # multiplicative noise on each person's effect
EFFECT_THETA_SLOPE = -0.10      # larger effects for lower baseline orientation
EFFECT_EDU = {"low": 1.1, "mid": 1.0, "high": 0.95}
EFFECT_AGE = {"18-29": 1.1, "30-44": 1.0, "45-59": 1.0, "60+": 0.9}

# sample size: every synthetic profile is simulated in ALL 17 conditions with common
# random numbers (exact counterfactuals), so each condition has N_PROFILES rows
# (floor is 500 / intervention, 1,000 control)
N_PROFILES = 4000
SEED = 20260831
