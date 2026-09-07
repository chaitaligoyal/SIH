"""
=============================================================================
GENERATES FICTIONAL DATA. NOT REAL INCIDENT REPORTS. DO NOT PRESENT AS REAL.
=============================================================================

These narratives were written to give the UI something to render before real
data was wired up. They are invented. They did not come from BSEE, PHMSA or
MSHA, and no regulator published them.

Use `backend/fetch_real_data.py` to pull actual regulator narratives. That
script downloads roughly 250,000 real MSHA accident narratives, which is what
should be in data/ for anything you submit or demo.

Keep these fixtures only for offline UI work and tests where hitting a
government server on every run would be silly. Every file this script writes
is prefixed `DEMO_FIXTURE_` so it is impossible to confuse with real data at
a glance, in a file listing, or in a screenshot.

The previous version of this file wrote `BSEE_Offshore_Incidents.csv`,
`PHMSA_Pipeline_Reports.csv` and `MSHA_Mining_Data.csv` — names indistinguishable
from real regulator exports — while the loader that read them was commented
"REAL-WORLD DATASET INGESTION". Delete those three files if they are still in
data/.
=============================================================================
"""

from __future__ import annotations

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

BANNER = "FICTIONAL — invented for UI testing, not a real incident report."

OFFSHORE = [
    "During wireline operations on the offshore platform, the riser tensioner line "
    "snapped due to corrosion, dropping heavy equipment near the wellbay.",
    "Mud pump pressure spiked unexpectedly during cementing operations, causing a "
    "blown seal and a minor blowout preventer fluid leak.",
    "Worker slipped on the drill floor during tripping operations; safety harness "
    "prevented a fall into the moonpool.",
    "Crane operator lost visibility during a heavy lift of drill collars in high "
    "winds, causing the load to swing and strike the derrick framework.",
    "Gas detector failed to alarm during a minor H2S release near the shale shakers. "
    "Manual override was required to trigger the platform stand-down.",
]

PIPELINE = [
    "Midstream pipeline pressure surge caused a flange gasket failure, leading to a "
    "hazardous material leak in the containment berm.",
    "During pigging operations, the receiver hatch was opened before complete "
    "depressurization, resulting in a sudden, high-velocity gas release.",
    "Excavator struck a marked underground crude line during trenching operations. "
    "No rupture, but the pipe coating was compromised.",
    "Energy isolation failure: block valve bypassed during compressor maintenance, "
    "allowing residual natural gas to enter the hot work zone.",
    "Corrosion anomaly on the mainline valve resulted in a slow hydrocarbon drip "
    "undetected by the automated SCADA pressure monitoring system.",
]

MINING = [
    "Haul truck brakes failed on a 5% grade. Operator steered into the runaway ramp, "
    "avoiding a collision with the heavy mobile plant.",
    "Ground control mesh gave way, causing a minor rockfall that struck the canopy of "
    "the continuous miner equipment.",
    "Mechanic attempted to clear a jammed conveyor belt without locking out the main "
    "drive motor. Belt unexpectedly shifted, trapping a tool.",
    "Hoist cable frayed and snapped while lifting a 2-ton replacement motor, dropping "
    "the load 10 feet.",
    "Arc flash occurred in the main electrical substation due to moisture ingress. "
    "Technician was wearing proper PPE, preventing severe burns.",
]


def write(name: str, prefix: str, narratives: list[str]) -> None:
    df = pd.DataFrame({
        "incident_id": [f"{prefix}-{i:03d}" for i in range(1, len(narratives) + 1)],
        "narrative": narratives,
        "provenance": [BANNER] * len(narratives),
    })
    path = os.path.join(DATA_DIR, name)
    df.to_csv(path, index=False)
    print(f"Wrote {path}  ({len(df)} fictional rows)")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 70)
    print("GENERATING FICTIONAL DEMO FIXTURES — these are not real incidents.")
    print("For real data run: python backend/fetch_real_data.py")
    print("=" * 70)
    write("DEMO_FIXTURE_offshore.csv", "FAKE-OFF", OFFSHORE)
    write("DEMO_FIXTURE_pipeline.csv", "FAKE-PIPE", PIPELINE)
    write("DEMO_FIXTURE_mining.csv", "FAKE-MINE", MINING)
    print("\nEvery row carries a `provenance` column marking it as fictional.")


if __name__ == "__main__":
    main()
