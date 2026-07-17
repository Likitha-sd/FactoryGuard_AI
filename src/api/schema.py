from pydantic import BaseModel, Field


class SensorInput(BaseModel):
    """
    Raw sensor inputs supplied by the user.

    All engineered features are computed automatically
    inside the inference pipeline.
    """

    # ======================================================
    # Machine Information
    # ======================================================

    machine_id: str = Field(
        ...,
        example="MACHINE_01",
        description="Unique Machine ID"
    )

    # ======================================================
    # Raw Sensor Features
    # ======================================================

    temperature_c: float = Field(
        ...,
        example=72.5,
        description="Temperature (°C)"
    )

    vibration_mm_s: float = Field(
        ...,
        example=3.5,
        description="Vibration (mm/s)"
    )

    pressure_kpa: float = Field(
        ...,
        example=120.0,
        description="Pressure (kPa)"
    )

    rotational_speed_rpm: float = Field(
        ...,
        example=1450,
        description="Rotational Speed (RPM)"
    )

    power_consumption_kw: float = Field(
        ...,
        example=11.5,
        description="Power Consumption (kW)"
    )

    operating_hours: float = Field(
        ...,
        example=2500,
        description="Operating Hours"
    )