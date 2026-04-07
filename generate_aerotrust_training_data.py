#!/usr/bin/env python3
"""
Generate synthetic aerospace maintenance discrepancy narratives for ModernBERT training.

Output schema:
  - raw_narrative
  - system_category: Avionics | Hydraulics | Structural | Propulsion | Cabin/ECS
  - risk_level: Low | Medium | Critical
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


SYSTEM_CATEGORIES = ["Avionics", "Hydraulics", "Structural", "Propulsion", "Cabin/ECS"]
RISK_LEVELS = ["Low", "Medium", "Critical"]
SCHEMA_COLUMNS = ["raw_narrative", "system_category", "risk_level"]

# Target distribution requested by user: ~60/30/10.
RISK_DISTRIBUTION = {"Low": 0.60, "Medium": 0.30, "Critical": 0.10}

FLIGHT_PHASES = [
    "DURING TAXI OUT",
    "DURING TAKEOFF ROLL",
    "IN CLIMB THROUGH 8,000 FT",
    "IN CLIMB THROUGH FL180",
    "AT CRUISE",
    "DURING DESCENT BELOW 10,000 FT",
    "ON APPROACH",
    "AFTER LANDING",
]

ENV_CONDITIONS = [
    "IN IMC WITH ANTI-ICE ON",
    "IN LIGHT TURBULENCE",
    "WITH PACKS IN AUTO",
    "WITH APU BLEED SELECTED",
    "DURING HIGH HUMIDITY CONDITIONS",
    "IN RAIN CONDITIONS",
    "WITH CABIN LOAD NEAR FULL",
]

AMM_BY_CATEGORY: Dict[str, List[str]] = {
    "Cabin/ECS": [
        "AMM 21-00-00",
        "AMM 21-22-00",
        "AMM 21-25-00",
        "AMM 21-51-11",
        "AMM 21-52-07",
        "AMM 44-22-43",
    ],
    "Propulsion": [
        "AMM 49-91-44",
        "AMM 72-00-00",
        "AMM 73-21-00",
        "AMM 74-11-01",
        "AMM 75-31-00",
    ],
    "Hydraulics": [
        "AMM 27-51-18",
        "AMM 29-11-15",
        "AMM 29-21-00",
        "AMM 29-31-00",
    ],
    "Avionics": [
        "AMM 23-00-00",
        "AMM 24-42-10",
        "AMM 31-31-00",
        "AMM 34-11-00",
        "AMM 34-21-00",
    ],
    "Structural": [
        "AMM 51-00-00",
        "AMM 52-11-00",
        "AMM 53-10-00",
        "AMM 53-21-00",
        "AMM 57-40-00",
    ],
}

PARTS_BY_CATEGORY: Dict[str, List[str]] = {
    "Cabin/ECS": [
        "PACK VALVE",
        "AIR CYCLE MACHINE",
        "RECIRCULATION FAN",
        "CABIN FILTER",
        "OUTLET UNIT",
        "MIX MANIFOLD DUCT",
    ],
    "Propulsion": [
        "OIL COOLER",
        "BLEED DUCT",
        "IGNITION LEAD",
        "FUEL NOZZLE",
        "ENGINE SENSOR HARNESS",
        "STARTER VALVE",
    ],
    "Hydraulics": [
        "POWER UNIT",
        "HYDRAULIC LINE",
        "ACTUATOR",
        "RETURN FILTER",
        "CASE DRAIN LINE",
        "SERVO VALVE",
    ],
    "Avionics": [
        "FMC MODULE",
        "DISPLAY UNIT",
        "ADC CHANNEL",
        "AUTOPILOT SERVO",
        "RADIO TUNING PANEL",
        "DATA CONCENTRATOR",
    ],
    "Structural": [
        "DOOR SEAL",
        "FLOOR PANEL",
        "FAIRING PANEL",
        "FASTENER ROW",
        "FRAME CLIP",
        "INSPECTION ACCESS PANEL",
    ],
}

FAULTS_BY_CATEGORY: Dict[str, List[str]] = {
    "Cabin/ECS": [
        "LEAKING",
        "STICKING",
        "INOPERATIVE",
        "CONTAMINATED",
        "OVERHEATING",
        "INTERMITTENT",
    ],
    "Propulsion": [
        "SEEPING",
        "CRACKED",
        "ARCED",
        "FAILED",
        "OUT OF LIMIT",
        "JAMMED",
    ],
    "Hydraulics": [
        "LEAKING",
        "SEIZED",
        "LOW OUTPUT",
        "INTERNAL BYPASS",
        "SLOW RESPONSE",
        "FAILED",
    ],
    "Avionics": [
        "FAULTED",
        "NO DATA",
        "ERRATIC",
        "FAILED BITE",
        "MISCOMPARE",
        "INOP",
    ],
    "Structural": [
        "CRACKED",
        "LOOSE",
        "DEFORMED",
        "WORN",
        "SEPARATED",
        "CORRODED",
    ],
}

SYMPTOMS_BY_CATEGORY: Dict[str, List[str]] = {
    "Cabin/ECS": [
        "DIRTY SOCK ODOR REPORTED FROM MID TO AFT CABIN",
        "BURNING RUBBER SMELL REPORTED BY FLIGHT ATTENDANT",
        "STALE AIR CONDITION WITH INTERMITTENT ODOR LEVEL 1",
        "CABIN TEMPERATURE CONTROL FLUCTUATION WITH ODOR EVENT",
    ],
    "Propulsion": [
        "OIL-LIKE ODOR REPORTED DURING DESCENT",
        "ENGINE VIBRATION TREND SHIFT NOTED BY CREW",
        "TRANSIENT EGT RISE DURING THRUST CHANGE",
        "FUEL-LIKE SMELL NOTED BRIEFLY AFTER TAKEOFF",
    ],
    "Hydraulics": [
        "HYD QUANTITY DROP INDICATION POST FLIGHT",
        "FLAP ASYMMETRY MESSAGE GENERATED IN APPROACH",
        "SLOW FLIGHT CONTROL RESPONSE REPORTED",
        "PRESSURE FLUCTUATION OBSERVED ON ECAM",
    ],
    "Avionics": [
        "INTERMITTENT AUTOPILOT DISCONNECT",
        "ADC DISAGREE MESSAGE ON PFD",
        "FMS POSITION JUMP REPORTED",
        "MULTIPLE AVIONICS BITE FAULTS AFTER POWER TRANSFER",
    ],
    "Structural": [
        "AIRFRAME VIBRATION/NOISE FROM FWD SECTION",
        "DOOR AREA WHISTLE AND PRESSURE LEAK INDICATION",
        "VISUAL DAMAGE FOUND DURING TRANSIT CHECK",
        "FASTENER WORKING EVIDENCE ON PANEL LINE",
    ],
}

FOLLOWUP_BY_RISK: Dict[str, List[str]] = {
    "Low": [
        "AIRCRAFT IS OK FOR SERVICE.",
        "DEFERRAL NOT REQUIRED; MONITOR NEXT TRANSIT CHECK.",
        "NO REPEAT AFTER OPS CHECK.",
        "NO ABNORMAL INDICATIONS REMAIN.",
    ],
    "Medium": [
        "REPAIR REQUIRED WITHIN NEXT 3 FLIGHT CYCLES.",
        "ITEM CONTROLLED FOR SHORT-TERM OPERATIONAL DISPATCH.",
        "FOLLOW-ON INSPECTION SCHEDULED WITHIN 24 HOURS.",
        "RESTRICTED RELEASE PENDING NEXT DAILY CHECK CONFIRMATION.",
    ],
    "Critical": [
        "AIRCRAFT PLACED AOG UNTIL RECTIFICATION COMPLETE.",
        "SAFETY-OF-FLIGHT IMPACT; IMMEDIATE CORRECTIVE ACTION TAKEN.",
        "DISPATCH NOT AUTHORIZED UNTIL FUNCTIONAL CHECK PASSED.",
        "CREW REPORTED SIGNIFICANT EFFECT; MAINTENANCE CONTROL NOTIFIED IMMEDIATELY.",
    ],
}

CORRECTIVE_ACTIONS = [
    "REMOVE AND REPLACE {part} PER {amm}.",
    "TROUBLESHOOT IAW {amm}; FOUND {part} {fault}.",
    "RESEATED/RETORQUED {part} PER {amm}.",
    "PERFORMED CLEANING, INSPECTION, AND FUNCTIONAL RESTORE PER {amm}.",
    "INSTALLED SERVICEABLE {part} AND CARRIED OUT OPS CHECK PER {amm}.",
]

CLOSEOUT_LINES = [
    "OPS CHECK GOOD.",
    "LEAK CHECK GOOD.",
    "FUNCTIONAL TEST SATISFACTORY.",
    "BITE TEST PASSED WITH NO FURTHER FAULTS.",
    "THIS CLEARS [MDDR].",
    "MX CONTROL UPDATED; PLACARDS REMOVED.",
]

SHORTHAND_TOKENS = [
    "[FLIGHT_DETAILS]",
    "[SYSTEM]",
    "[ESPM_REF]",
    "[MEL_REF]",
    "[MDDR]",
]

CATEGORY_WEIGHTS = {
    "Cabin/ECS": 0.33,
    "Propulsion": 0.20,
    "Hydraulics": 0.18,
    "Avionics": 0.17,
    "Structural": 0.12,
}


@dataclass(frozen=True)
class Scenario:
    system_category: str
    risk_level: str
    part: str
    fault: str
    symptom: str
    amm: str
    phase: str
    condition: str
    token: str


def weighted_choice(weights: Dict[str, float]) -> str:
    labels = list(weights.keys())
    probs = list(weights.values())
    return random.choices(labels, probs, k=1)[0]


def build_risk_pool(total_rows: int) -> List[str]:
    exact_counts = {
        risk: int(total_rows * ratio) for risk, ratio in RISK_DISTRIBUTION.items()
    }
    remainder = total_rows - sum(exact_counts.values())

    # Assign remainder deterministically by largest target ratio.
    ordered = sorted(RISK_DISTRIBUTION.items(), key=lambda x: x[1], reverse=True)
    idx = 0
    while remainder > 0:
        exact_counts[ordered[idx][0]] += 1
        idx = (idx + 1) % len(ordered)
        remainder -= 1

    pool: List[str] = []
    for risk in RISK_LEVELS:
        pool.extend([risk] * exact_counts[risk])
    random.shuffle(pool)
    return pool


def choose_fault_for_risk(faults: List[str], risk_level: str) -> str:
    if risk_level == "Critical":
        return random.choice(faults[-3:] if len(faults) >= 3 else faults)
    if risk_level == "Low":
        return random.choice(faults[:3] if len(faults) >= 3 else faults)
    return random.choice(faults)


def build_scenario(risk_level: str) -> Scenario:
    category = weighted_choice(CATEGORY_WEIGHTS)
    parts = PARTS_BY_CATEGORY[category]
    faults = FAULTS_BY_CATEGORY[category]
    symptoms = SYMPTOMS_BY_CATEGORY[category]
    amm_refs = AMM_BY_CATEGORY[category]

    return Scenario(
        system_category=category,
        risk_level=risk_level,
        part=random.choice(parts),
        fault=choose_fault_for_risk(faults, risk_level),
        symptom=random.choice(symptoms),
        amm=random.choice(amm_refs),
        phase=random.choice(FLIGHT_PHASES),
        condition=random.choice(ENV_CONDITIONS),
        token=random.choice(SHORTHAND_TOKENS),
    )


def build_narrative(s: Scenario) -> str:
    intro = (
        f"{s.token} {s.symptom}. EVENT OCCURRED {s.phase} {s.condition}. "
        f"COMPLIED WITH INFORMATION NOTICE 2017-T015F SECTION "
        f"{random.choice(['A', 'B', 'C'])}."
    )
    finding = f"FOUND {s.part} {s.fault}."
    action_template = random.choice(CORRECTIVE_ACTIONS)
    action = action_template.format(part=s.part, amm=s.amm, fault=s.fault)
    followup = random.choice(FOLLOWUP_BY_RISK[s.risk_level])

    closeout_counts = {"Low": 1, "Medium": 2, "Critical": 3}
    closeout_count = closeout_counts[s.risk_level]
    closeouts = " ".join(random.sample(CLOSEOUT_LINES, k=closeout_count))

    return f"{intro} {finding} {action} {followup} {closeouts}"


def generate_dataset(row_count: int, seed: int | None = None) -> pd.DataFrame:
    if row_count < 1000:
        raise ValueError("row_count must be >= 1000 to satisfy the 1,000+ requirement.")
    if seed is not None:
        random.seed(seed)

    risk_pool = build_risk_pool(row_count)
    records = []
    for risk_level in risk_pool:
        scenario = build_scenario(risk_level)
        records.append(
            {
                "raw_narrative": build_narrative(scenario),
                "system_category": scenario.system_category,
                "risk_level": scenario.risk_level,
            }
        )

    return pd.DataFrame(records, columns=SCHEMA_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic AeroTrust maintenance narratives for classifier training."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1200,
        help="Number of rows to generate (must be >= 1000). Default: 1200",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aerotrust_training_data.csv",
        help="Output CSV path. Default: aerotrust_training_data.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_dataset(row_count=args.rows, seed=args.seed)
    df.to_csv(args.output, index=False)

    risk_percentages = (df["risk_level"].value_counts(normalize=True) * 100).round(1)
    counts = {risk: float(risk_percentages.get(risk, 0.0)) for risk in RISK_LEVELS}
    print(f"Generated {len(df)} rows -> {args.output}")
    print(f"Risk distribution (%): {counts}")
    print(f"Columns: {SCHEMA_COLUMNS}")


if __name__ == "__main__":
    main()
