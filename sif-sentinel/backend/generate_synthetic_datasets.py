import pandas as pd
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# 1. BSEE Synthetic Data (Offshore Oil & Gas)
bsee_data = {
    "incident_id": ["BSEE-001", "BSEE-002", "BSEE-003", "BSEE-004", "BSEE-005"],
    "description": [
        "During wireline operations on the offshore platform, the riser tensioner line snapped due to corrosion, dropping heavy equipment near the wellbay.",
        "Mud pump pressure spiked unexpectedly during cementing operations, causing a blown seal and a minor blowout preventer (BOP) fluid leak.",
        "Worker slipped on the drill floor during tripping operations; safety harness prevented a fall into the moonpool.",
        "Crane operator lost visibility during a heavy lift of drill collars in high winds, causing the load to swing and strike the derrick framework.",
        "Gas detector failed to alarm during a minor H2S release near the shale shakers. Manual override was required to trigger the platform stand-down."
    ]
}
pd.DataFrame(bsee_data).to_csv("data/BSEE_Offshore_Incidents.csv", index=False)
print("Generated data/BSEE_Offshore_Incidents.csv")

# 2. PHMSA Synthetic Data (Pipeline & Transport)[cite: 2]
phmsa_data = {
    "report_id": ["PHMSA-101", "PHMSA-102", "PHMSA-103", "PHMSA-104", "PHMSA-105"],
    "narrative_text": [
        "Midstream pipeline pressure surge caused a flange gasket failure, leading to a hazardous material leak in the containment berm.",
        "During pigging operations, the receiver hatch was opened before complete depressurization, resulting in a sudden, high-velocity gas release.",
        "Excavator struck a marked underground crude line during trenching operations. No rupture, but structural integrity of the pipe coating was compromised.",
        "Energy isolation failure: Block valve bypassed during compressor maintenance, allowing residual natural gas to enter the hot work zone.",
        "Corrosion anomaly on the mainline valve resulted in a slow hydrocarbon drip undetected by the automated SCADA pressure monitoring system."
    ]
}
pd.DataFrame(phmsa_data).to_csv("data/PHMSA_Pipeline_Reports.csv", index=False)
print("Generated data/PHMSA_Pipeline_Reports.csv")

# 3. MSHA Synthetic Data (Mine / Heavy Industry)[cite: 2]
msha_data = {
    "id": ["MSHA-901", "MSHA-902", "MSHA-903", "MSHA-904", "MSHA-905"],
    "text": [
        "Haul truck brakes failed on a 5% grade. Operator steered into the runaway ramp, avoiding a collision with the heavy mobile plant.",
        "Structural failure: Ground control mesh gave way, causing a minor rockfall that struck the canopy of the continuous miner equipment.",
        "Mechanic attempted to clear a jammed conveyor belt without locking out the main drive motor. Belt unexpectedly shifted, trapping a tool.",
        "Suspended load hazard: Hoist cable frayed and snapped while lifting a 2-ton replacement motor, dropping the load 10 feet.",
        "Arc flash occurred in the main electrical substation due to moisture ingress. Technician was wearing proper PPE, preventing severe burns."
    ]
}
pd.DataFrame(msha_data).to_csv("data/MSHA_Mining_Data.csv", index=False)
print("Generated data/MSHA_Mining_Data.csv")