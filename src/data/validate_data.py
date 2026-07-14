"""
Validate the raw FactoryGuard industrial sensor dataset.

Checks:
- required columns
- dataset shape
- duplicate rows
- missing values
- target validity
- numeric sensor ranges
- timestamp validity
- machine identifiers
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


# ============================================================
# EXPECTED DATA SCHEMA
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    "temperature_c",
    "vibration_mm_s",
    "pressure_kpa",
    "rotational_speed_rpm",
    "power_consumption_kw",
    "operating_hours",
    "machine_failure",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data() -> pd.DataFrame:
    """Load the FactoryGuard raw sensor dataset."""

    if not RAW_DATA_FILE.exists():

        raise FileNotFoundError(
            f"Raw dataset was not found:\n{RAW_DATA_FILE}"
        )

    data = pd.read_csv(
        RAW_DATA_FILE
    )

    return data


# ============================================================
# VALIDATION
# ============================================================

def validate_factory_data(
    data: pd.DataFrame,
) -> bool:
    """
    Validate dataset structure and data quality.

    Returns
    -------
    bool
        True when every validation check passes.
    """

    validation_passed = True

    print("=" * 70)
    print("FACTORYGUARD DATA VALIDATION REPORT")
    print("=" * 70)

    print(
        "Dataset shape:",
        data.shape,
    )

    print(
        "Rows         :",
        len(data),
    )

    print(
        "Columns      :",
        len(data.columns),
    )

    print("-" * 70)


    # --------------------------------------------------------
    # 1. REQUIRED COLUMNS
    # --------------------------------------------------------

    missing_columns = [

        column

        for column in REQUIRED_COLUMNS

        if column not in data.columns

    ]

    if missing_columns:

        validation_passed = False

        print(
            "Required columns : FAILED ❌"
        )

        print(
            "Missing columns  :",
            missing_columns,
        )

    else:

        print(
            "Required columns : PASSED ✅"
        )


    # --------------------------------------------------------
    # 2. DUPLICATE ROWS
    # --------------------------------------------------------

    duplicate_count = int(
        data.duplicated().sum()
    )

    print(
        "Duplicate rows   :",
        duplicate_count,
    )

    if duplicate_count == 0:

        print(
            "Duplicate check  : PASSED ✅"
        )

    else:

        validation_passed = False

        print(
            "Duplicate check  : FAILED ❌"
        )


    # --------------------------------------------------------
    # 3. MISSING VALUES
    # --------------------------------------------------------

    missing_values = (
        data
        .isna()
        .sum()
    )

    total_missing_values = int(
        missing_values.sum()
    )

    print(
        "Missing values   :",
        total_missing_values,
    )

    if total_missing_values == 0:

        print(
            "Missing check    : PASSED ✅"
        )

    else:

        validation_passed = False

        print(
            "Missing check    : FAILED ❌"
        )

        print(
            missing_values[
                missing_values > 0
            ]
        )


    # --------------------------------------------------------
    # 4. TARGET VALUES
    # --------------------------------------------------------

    if "machine_failure" in data.columns:

        target_values = set(

            data[
                "machine_failure"
            ]
            .dropna()
            .unique()

        )

        expected_target_values = {
            0,
            1,
        }

        if target_values.issubset(
            expected_target_values
        ):

            print(
                "Target values    : PASSED ✅"
            )

        else:

            validation_passed = False

            print(
                "Target values    : FAILED ❌"
            )

            print(
                "Unexpected values:",
                target_values,
            )


    # --------------------------------------------------------
    # 5. NUMERIC SENSOR RANGES
    # --------------------------------------------------------

    numeric_columns = [
        "temperature_c",
        "vibration_mm_s",
        "pressure_kpa",
        "rotational_speed_rpm",
        "power_consumption_kw",
        "operating_hours",
    ]

    invalid_numeric_values = 0

    for column in numeric_columns:

        if column in data.columns:

            invalid_numeric_values += int(

                (
                    data[column]
                    < 0
                )
                .sum()

            )

    print(
        "Negative sensor values:",
        invalid_numeric_values,
    )

    if invalid_numeric_values == 0:

        print(
            "Sensor ranges     : PASSED ✅"
        )

    else:

        validation_passed = False

        print(
            "Sensor ranges     : FAILED ❌"
        )


    # --------------------------------------------------------
    # 6. TIMESTAMP VALIDITY
    # --------------------------------------------------------

    if "timestamp" in data.columns:

        parsed_timestamps = pd.to_datetime(

            data[
                "timestamp"
            ],

            errors="coerce",

        )

        invalid_timestamps = int(

            parsed_timestamps
            .isna()
            .sum()

        )

        print(
            "Invalid timestamps:",
            invalid_timestamps,
        )

        if invalid_timestamps == 0:

            print(
                "Timestamp check   : PASSED ✅"
            )

        else:

            validation_passed = False

            print(
                "Timestamp check   : FAILED ❌"
            )


    # --------------------------------------------------------
    # 7. MACHINE IDENTIFIERS
    # --------------------------------------------------------

    if "machine_id" in data.columns:

        machine_count = int(

            data[
                "machine_id"
            ]
            .nunique()

        )

        print(
            "Unique machines   :",
            machine_count,
        )

        if machine_count > 0:

            print(
                "Machine ID check  : PASSED ✅"
            )

        else:

            validation_passed = False

            print(
                "Machine ID check  : FAILED ❌"
            )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("=" * 70)

    if validation_passed:

        print(
            "FINAL RESULT: ALL VALIDATION CHECKS PASSED ✅"
        )

    else:

        print(
            "FINAL RESULT: DATA VALIDATION FAILED ❌"
        )

    print("=" * 70)

    return validation_passed


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Load and validate the raw FactoryGuard dataset."""

    factory_data = load_raw_data()

    validation_passed = validate_factory_data(
        factory_data
    )

    if not validation_passed:

        raise ValueError(
            "FactoryGuard raw dataset failed validation."
        )


if __name__ == "__main__":
    main()