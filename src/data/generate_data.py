"""
Generate a reproducible industrial machine-sensor dataset
for the FactoryGuard AI project.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = RAW_DATA_DIR / "factory_sensor_data.csv"


def generate_factory_data(
    number_of_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate industrial machine-sensor observations.

    Parameters
    ----------
    number_of_samples:
        Number of sensor observations to generate.

    random_state:
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Factory sensor dataset with a binary failure label.
    """

    rng = np.random.default_rng(
        random_state
    )

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=number_of_samples,
        freq="min",
    )

    machine_ids = rng.choice(
        [
            "MACHINE_01",
            "MACHINE_02",
            "MACHINE_03",
            "MACHINE_04",
            "MACHINE_05",
        ],
        size=number_of_samples,
    )

    temperature = rng.normal(
        loc=72,
        scale=8,
        size=number_of_samples,
    )

    vibration = rng.normal(
        loc=2.8,
        scale=0.7,
        size=number_of_samples,
    )

    pressure = rng.normal(
        loc=101,
        scale=7,
        size=number_of_samples,
    )

    rotational_speed = rng.normal(
        loc=1500,
        scale=180,
        size=number_of_samples,
    )

    power_consumption = rng.normal(
        loc=52,
        scale=9,
        size=number_of_samples,
    )

    operating_hours = rng.uniform(
        low=100,
        high=12000,
        size=number_of_samples,
    )

    failure_score = (
        0.08 * (temperature - 72)
        + 0.95 * (vibration - 2.8)
        - 0.05 * (pressure - 101)
        + 0.0015 * (rotational_speed - 1500)
        + 0.035 * (power_consumption - 52)
        + 0.00015 * (operating_hours - 6000)
        + rng.normal(
            loc=0,
            scale=0.7,
            size=number_of_samples,
        )
    )

    failure_threshold = np.quantile(
        failure_score,
        0.90,
    )

    machine_failure = (
        failure_score >= failure_threshold
    ).astype(int)

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "machine_id": machine_ids,
            "temperature_c": temperature,
            "vibration_mm_s": vibration,
            "pressure_kpa": pressure,
            "rotational_speed_rpm": rotational_speed,
            "power_consumption_kw": power_consumption,
            "operating_hours": operating_hours,
            "machine_failure": machine_failure,
        }
    )

    return data


def main() -> None:
    """Generate and save the raw dataset."""

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    factory_data = generate_factory_data()

    factory_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 65)
    print("FACTORYGUARD RAW DATA CREATED")
    print("=" * 65)

    print(
        "Dataset path :",
        OUTPUT_FILE,
    )

    print(
        "Dataset shape:",
        factory_data.shape,
    )

    print(
        "Failure count:",
        int(
            factory_data[
                "machine_failure"
            ].sum()
        ),
    )

    print(
        "Normal count :",
        int(
            (
                factory_data[
                    "machine_failure"
                ]
                == 0
            ).sum()
        ),
    )

    print("=" * 65)


if __name__ == "__main__":
    main()