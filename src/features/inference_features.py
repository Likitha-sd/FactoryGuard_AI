import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "machine_statistics.csv"
)

# Load machine statistics once
MACHINE_STATS = pd.read_csv(STATS_FILE)


def _get_machine_stats(machine_id):
    row = MACHINE_STATS[MACHINE_STATS["machine_id"] == machine_id]

    if row.empty:
        raise ValueError(f"Machine '{machine_id}' not found in machine_statistics.csv")

    return row.iloc[0]


def build_inference_features(sensor_data: dict):
    """
    Converts raw sensor inputs into the exact feature set
    expected by the trained Random Forest model.
    """

    data = sensor_data.copy()

    # ---------------------------------------------------
    # Timestamp Features
    # ---------------------------------------------------

    now = datetime.now()

    data["hour"] = now.hour
    data["day_of_week"] = now.weekday()

    data["is_weekend"] = int(data["day_of_week"] >= 5)

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    # ---------------------------------------------------
    # Interaction Features
    # ---------------------------------------------------

    data["thermal_load"] = (
        data["temperature_c"]
        * data["power_consumption_kw"]
    )

    data["vibration_speed_interaction"] = (
        data["vibration_mm_s"]
        * data["rotational_speed_rpm"]
    )

    data["power_per_1000_rpm"] = (
        data["power_consumption_kw"]
        / (data["rotational_speed_rpm"] / 1000 + 1e-6)
    )

    data["pressure_temperature_ratio"] = (
        data["pressure_kpa"]
        / (data["temperature_c"] + 1e-6)
    )

    # ---------------------------------------------------
    # Machine Statistics
    # ---------------------------------------------------

    stats = _get_machine_stats(data["machine_id"])

    sensors = [
        "temperature_c",
        "vibration_mm_s",
        "pressure_kpa",
        "rotational_speed_rpm",
        "power_consumption_kw",
        "operating_hours",
    ]

    for sensor in sensors:

        mean = stats[f"{sensor}_mean"]
        std = stats[f"{sensor}_std"]

        if std == 0:
            std = 1e-6

        data[f"{sensor}_machine_zscore"] = (
            data[sensor] - mean
        ) / std

    # ---------------------------------------------------
    # Composite Stress Indicators
    # ---------------------------------------------------

    data["thermal_stress_index"] = (
        data["temperature_c"]
        * data["power_consumption_kw"]
    )

    data["mechanical_stress_index"] = (
        data["vibration_mm_s"]
        * data["rotational_speed_rpm"]
    )

    data["overall_stress_index"] = (
        data["thermal_stress_index"]
        + data["mechanical_stress_index"]
    ) / 2

    # ---------------------------------------------------
    # Final Feature Order
    # ---------------------------------------------------

    feature_order = [
        "machine_id",
        "temperature_c",
        "vibration_mm_s",
        "pressure_kpa",
        "rotational_speed_rpm",
        "power_consumption_kw",
        "operating_hours",
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "thermal_load",
        "vibration_speed_interaction",
        "power_per_1000_rpm",
        "pressure_temperature_ratio",
        "temperature_c_machine_zscore",
        "vibration_mm_s_machine_zscore",
        "pressure_kpa_machine_zscore",
        "rotational_speed_rpm_machine_zscore",
        "power_consumption_kw_machine_zscore",
        "operating_hours_machine_zscore",
        "thermal_stress_index",
        "mechanical_stress_index",
        "overall_stress_index",
    ]

    df = pd.DataFrame([data])

    return df[feature_order]


if __name__ == "__main__":

    sample = {
        "machine_id": "M001",
        "temperature_c": 74.5,
        "vibration_mm_s": 3.2,
        "pressure_kpa": 125.0,
        "rotational_speed_rpm": 1450,
        "power_consumption_kw": 11.5,
        "operating_hours": 2150,
    }

    features = build_inference_features(sample)

    print("=" * 70)
    print(features)
    print("=" * 70)