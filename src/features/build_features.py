"""
Build machine-health features for FactoryGuard AI.

Feature groups:
1. Time-based operating features
2. Sensor interaction features
3. Machine-relative deviation features
4. Composite stress indicators

The target column is preserved but is never used to construct features.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "factory_sensor_data_clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "factory_sensor_features.csv"
)


# ============================================================
# ORIGINAL SENSOR COLUMNS
# ============================================================

SENSOR_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_kpa",
    "rotational_speed_rpm",
    "power_consumption_kw",
    "operating_hours",
]


# ============================================================
# LOAD CLEAN DATA
# ============================================================

def load_processed_data() -> pd.DataFrame:
    """Load the cleaned FactoryGuard sensor dataset."""

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Processed dataset was not found:\n{INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    return data


# ============================================================
# BUILD FEATURES
# ============================================================

def build_factory_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create machine-health features without target leakage.

    Parameters
    ----------
    data:
        Clean FactoryGuard sensor dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset containing original and engineered features.
    """

    featured_data = data.copy()


    # --------------------------------------------------------
    # 1. PARSE TIMESTAMP
    # --------------------------------------------------------

    featured_data[
        "timestamp"
    ] = pd.to_datetime(

        featured_data[
            "timestamp"
        ],

        errors="raise",

    )


    # --------------------------------------------------------
    # 2. TIME FEATURES
    # --------------------------------------------------------

    featured_data[
        "hour"
    ] = (

        featured_data[
            "timestamp"
        ]
        .dt
        .hour

    )


    featured_data[
        "day_of_week"
    ] = (

        featured_data[
            "timestamp"
        ]
        .dt
        .dayofweek

    )


    featured_data[
        "is_weekend"
    ] = (

        featured_data[
            "day_of_week"
        ]

        .isin(
            [5, 6]
        )

        .astype(
            int
        )

    )


    # Cyclical hour representation

    featured_data[
        "hour_sin"
    ] = np.sin(

        2

        *

        np.pi

        *

        featured_data[
            "hour"
        ]

        /

        24

    )


    featured_data[
        "hour_cos"
    ] = np.cos(

        2

        *

        np.pi

        *

        featured_data[
            "hour"
        ]

        /

        24

    )


    # --------------------------------------------------------
    # 3. SENSOR INTERACTION FEATURES
    # --------------------------------------------------------

    featured_data[
        "thermal_load"
    ] = (

        featured_data[
            "temperature_c"
        ]

        *

        featured_data[
            "power_consumption_kw"
        ]

    )


    featured_data[
        "vibration_speed_interaction"
    ] = (

        featured_data[
            "vibration_mm_s"
        ]

        *

        featured_data[
            "rotational_speed_rpm"
        ]

    )


    featured_data[
        "power_per_1000_rpm"
    ] = (

        featured_data[
            "power_consumption_kw"
        ]

        /

        (

            featured_data[
                "rotational_speed_rpm"
            ]

            /

            1000

            +

            1e-8

        )

    )


    featured_data[
        "pressure_temperature_ratio"
    ] = (

        featured_data[
            "pressure_kpa"
        ]

        /

        (

            featured_data[
                "temperature_c"
            ]

            +

            1e-8

        )

    )


    # --------------------------------------------------------
    # 4. MACHINE-RELATIVE SENSOR DEVIATIONS
    # --------------------------------------------------------

    for sensor in SENSOR_COLUMNS:


        machine_mean = (

            featured_data

            .groupby(
                "machine_id"
            )[sensor]

            .transform(
                "mean"
            )

        )


        machine_standard_deviation = (

            featured_data

            .groupby(
                "machine_id"
            )[sensor]

            .transform(
                "std"
            )

            .replace(
                0,
                1
            )

            .fillna(
                1
            )

        )


        featured_data[
            f"{sensor}_machine_zscore"
        ] = (

            featured_data[
                sensor
            ]

            -

            machine_mean

        ) / (

            machine_standard_deviation

            +

            1e-8

        )


    # --------------------------------------------------------
    # 5. COMPOSITE MACHINE-STRESS FEATURES
    # --------------------------------------------------------

    featured_data[
        "thermal_stress_index"
    ] = (

        featured_data[
            "temperature_c_machine_zscore"
        ]

        +

        featured_data[
            "power_consumption_kw_machine_zscore"
        ]

    ) / 2


    featured_data[
        "mechanical_stress_index"
    ] = (

        featured_data[
            "vibration_mm_s_machine_zscore"
        ]

        +

        featured_data[
            "rotational_speed_rpm_machine_zscore"
        ]

    ) / 2


    featured_data[
        "overall_stress_index"
    ] = (

        featured_data[
            "thermal_stress_index"
        ]

        +

        featured_data[
            "mechanical_stress_index"
        ]

        -

        featured_data[
            "pressure_kpa_machine_zscore"
        ]

    ) / 3


    # --------------------------------------------------------
    # 6. SORT DATA
    # --------------------------------------------------------

    featured_data = (

        featured_data

        .sort_values(

            by=[
                "timestamp",
                "machine_id",
            ]

        )

        .reset_index(
            drop=True
        )

    )


    return featured_data


# ============================================================
# SAVE FEATURES
# ============================================================

def save_feature_data(
    featured_data: pd.DataFrame,
) -> None:
    """Save the feature-engineered dataset."""

    OUTPUT_FILE.parent.mkdir(

        parents=True,

        exist_ok=True,

    )


    featured_data.to_csv(

        OUTPUT_FILE,

        index=False,

    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Execute the FactoryGuard feature pipeline."""

    print("=" * 72)
    print("FACTORYGUARD FEATURE ENGINEERING")
    print("=" * 72)


    processed_data = load_processed_data()


    original_column_count = len(
        processed_data.columns
    )


    featured_data = build_factory_features(
        processed_data
    )


    save_feature_data(
        featured_data
    )


    engineered_column_count = (

        len(
            featured_data.columns
        )

        -

        original_column_count

    )


    print(
        "Input shape             :",
        processed_data.shape,
    )


    print(
        "Feature dataset shape   :",
        featured_data.shape,
    )


    print(
        "Original columns        :",
        original_column_count,
    )


    print(
        "New engineered features :",
        engineered_column_count,
    )


    print(
        "Missing values          :",
        int(

            featured_data

            .isna()

            .sum()

            .sum()

        ),
    )


    print(
        "Infinite numeric values :",
        int(

            np.isinf(

                featured_data

                .select_dtypes(
                    include=np.number
                )

                .to_numpy()

            )

            .sum()

        ),
    )


    print(
        "Output file             :",
        OUTPUT_FILE,
    )


    print("-" * 72)

    print(
        "New feature names:"
    )


    new_features = [

        column

        for column in featured_data.columns

        if column not in processed_data.columns

    ]


    for feature in new_features:

        print(
            f"  + {feature}"
        )


    print("=" * 72)

    print(
        "FACTORYGUARD FEATURE ENGINEERING COMPLETED ✅"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()