from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from src.api.schema import SensorInput
from src.models.predict import predict_failure

from src.database.database import get_db
from src.database.crud import save_prediction

# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title="FactoryGuard AI",
    version="2.0",
    description="Industrial Predictive Maintenance API"
)

# ==========================================================
# Home Route
# ==========================================================

@app.get("/")
def home():
    return {
        "project": "FactoryGuard AI",
        "status": "Running",
        "version": "2.0",
        "message": "Industrial Predictive Maintenance API"
    }


# ==========================================================
# Health Check
# ==========================================================

@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }


# ==========================================================
# Prediction Route
# ==========================================================

@app.post("/predict")
def predict(
    sensor_data: SensorInput,
    db: Session = Depends(get_db)
):

    try:

        result = predict_failure(sensor_data.model_dump())

        # --------------------------------------------------
        # Save prediction to SQLite
        # --------------------------------------------------

        save_prediction(
            db=db,
            machine_id=sensor_data.machine_id,
            temperature=sensor_data.temperature_c,
            vibration=sensor_data.vibration_mm_s,
            pressure=sensor_data.pressure_kpa,
            rpm=sensor_data.rotational_speed_rpm,
            power=sensor_data.power_consumption_kw,
            operating_hours=sensor_data.operating_hours,
            prediction=result["prediction"],
            probability=result["failure_probability"],
        )

        return {
            "success": True,
            "prediction": result["prediction"],
            "failure_probability": result["failure_probability"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )