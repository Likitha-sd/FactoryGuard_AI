import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "factory_sensor_features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "machine_statistics.csv"

SENSORS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_kpa",
    "rotational_speed_rpm",
    "power_consumption_kw",
    "operating_hours",
]

df = pd.read_csv(INPUT_FILE)

rows = []

for machine in df["machine_id"].unique():

    row = {"machine_id": machine}

    machine_df = df[df["machine_id"] == machine]

    for sensor in SENSORS:

        row[f"{sensor}_mean"] = machine_df[sensor].mean()
        row[f"{sensor}_std"] = machine_df[sensor].std()

    rows.append(row)

stats = pd.DataFrame(rows)

stats.to_csv(OUTPUT_FILE, index=False)

print(stats.head())

print("\nSaved to:")

print(OUTPUT_FILE)