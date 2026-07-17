from sqlalchemy.orm import Session

from .models import Prediction


def save_prediction(
    db,
    machine_id,
    temperature,
    vibration,
    pressure,
    rpm,
    power,
    operating_hours,
    prediction,
    probability,
):

    row = Prediction(
        machine_id=machine_id,
        temperature=temperature,
        vibration=vibration,
        pressure=pressure,
        rpm=rpm,
        power=power,
        operating_hours=operating_hours,
        prediction=prediction,
        probability=probability,
    )

    db.add(row)
    db.commit()
    db.refresh(row)

    return row


def get_prediction_history(db: Session):

    return (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .all()
    )