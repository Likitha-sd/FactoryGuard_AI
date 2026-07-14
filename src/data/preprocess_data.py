"""
Preprocess the validated FactoryGuard sensor dataset.

Processing steps:
1. Load raw sensor data.
2. Parse timestamps.
3. Sort observations chronologically.
4. Remove duplicate rows.
5. Handle missing values.
6. Validate numeric sensor columns.
7. Save a clean processed dataset.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "factory_sensor_data.csv"
)

PROCESSED_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIRECTORY
    / "factory_sensor_data_clean.csv"
)


# ============================================================
# DATA COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_kpa",
    "rotational_speed_rpm",
    "power_consumption_kw",
    "operating_hours",
]

REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    *NUMERIC_COLUMNS,
    "machine_failure",
]


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_data() -> pd.DataFrame:
    """Load the validated raw FactoryGuard dataset."""

    if not RAW_DATA_FILE.exists():

        raise FileNotFoundError(
            f"Raw dataset was not found:\n{RAW_DATA_FILE}"
        )

    data = pd.read_csv(
        RAW_DATA_FILE
    )

    return data


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

def verify_required_columns(
    data: pd.DataFrame,
) -> None:
    """Verify that every required column exists."""

    missing_columns = [

        column

        for column in REQUIRED_COLUMNS

        if column not in data.columns

    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_factory_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and prepare the FactoryGuard sensor dataset.

    Parameters
    ----------
    data:
        Raw industrial sensor dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned and chronologically ordered dataset.
    """

    processed_data = data.copy()

    verify_required_columns(
        processed_data
    )


    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    processed_data[
        "timestamp"
    ] = pd.to_datetime(

        processed_data[
            "timestamp"
        ],

        errors="coerce",

    )


    # --------------------------------------------------------
    # Convert sensor columns to numeric values
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        processed_data[
            column
        ] = pd.to_numeric(

            processed_data[
                column
            ],

            errors="coerce",

        )


    # --------------------------------------------------------
    # Convert target to numeric values
    # --------------------------------------------------------

    processed_data[
        "machine_failure"
    ] = pd.to_numeric(

        processed_data[
            "machine_failure"
        ],

        errors="coerce",

    )


    # --------------------------------------------------------
    # Remove duplicate observations
    # --------------------------------------------------------

    processed_data = (

        processed_data

        .drop_duplicates()

        .copy()

    )


    # --------------------------------------------------------
    # Remove rows missing identifiers or target values
    # --------------------------------------------------------

    processed_data = (

        processed_data

        .dropna(

            subset=[
                "timestamp",
                "machine_id",
                "machine_failure",
            ]

        )

        .copy()

    )


    # --------------------------------------------------------
    # Fill missing sensor values using machine-level medians
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        machine_medians = (

            processed_data

            .groupby(
                "machine_id"
            )[column]

            .transform(
                "median"
            )

        )


        processed_data[
            column
        ] = (

            processed_data[
                column
            ]

            .fillna(
                machine_medians
            )

        )


        processed_data[
            column
        ] = (

            processed_data[
                column
            ]

            .fillna(

                processed_data[
                    column
                ]
                .median()

            )

        )


    # --------------------------------------------------------
    # Remove physically impossible negative sensor values
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        processed_data = (

            processed_data[

                processed_data[
                    column
                ]
                >= 0

            ]

            .copy()

        )


    # --------------------------------------------------------
    # Standardize machine identifiers
    # --------------------------------------------------------

    processed_data[
        "machine_id"
    ] = (

        processed_data[
            "machine_id"
        ]

        .astype(
            str
        )

        .str.strip()

        .str.upper()

    )


    # --------------------------------------------------------
    # Convert target to integer
    # --------------------------------------------------------

    processed_data[
        "machine_failure"
    ] = (

        processed_data[
            "machine_failure"
        ]

        .astype(
            int
        )

    )


    # --------------------------------------------------------
    # Sort observations chronologically
    # --------------------------------------------------------

    processed_data = (

        processed_data

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


    return processed_data


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    processed_data: pd.DataFrame,
) -> None:
    """Save the clean FactoryGuard dataset."""

    PROCESSED_DATA_DIRECTORY.mkdir(

        parents=True,

        exist_ok=True,

    )

    processed_data.to_csv(

        PROCESSED_DATA_FILE,

        index=False,

    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Execute the FactoryGuard preprocessing pipeline."""

    print("=" * 70)
    print("FACTORYGUARD DATA PREPROCESSING")
    print("=" * 70)


    raw_data = load_raw_data()


    print(
        "Raw dataset shape      :",
        raw_data.shape,
    )


    processed_data = preprocess_factory_data(
        raw_data
    )


    save_processed_data(
        processed_data
    )


    print(
        "Processed dataset shape:",
        processed_data.shape,
    )


    print(
        "Remaining missing values:",
        int(
            processed_data
            .isna()
            .sum()
            .sum()
        ),
    )


    print(
        "Remaining duplicates   :",
        int(
            processed_data
            .duplicated()
            .sum()
        ),
    )


    print(
        "Failure observations   :",
        int(
            processed_data[
                "machine_failure"
            ]
            .sum()
        ),
    )


    print(
        "Normal observations    :",
        int(

            (
                processed_data[
                    "machine_failure"
                ]
                == 0
            )

            .sum()

        ),
    )


    print(
        "Processed file         :",
        PROCESSED_DATA_FILE,
    )


    print("=" * 70)
    print("FACTORYGUARD PREPROCESSING COMPLETED ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()