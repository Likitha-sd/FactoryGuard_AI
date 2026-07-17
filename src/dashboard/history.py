import pandas as pd

from src.database.database import SessionLocal
from src.dashboard.analytics import load_dashboard_data


def get_history_dataframe():

    db = SessionLocal()

    data = load_dashboard_data(db)

    rows = []

    for row in data["history"]:

        rows.append(
            {
                "Timestamp": row.timestamp,
                "Machine": row.machine_id,
                "Prediction": row.prediction,
                "Failure Probability": round(row.probability, 4),
                "Temperature": row.temperature,
                "Vibration": row.vibration,
                "Pressure": row.pressure,
                "RPM": row.rpm,
                "Power": row.power,
                "Operating Hours": row.operating_hours,
            }
        )

    db.close()

    return pd.DataFrame(rows)