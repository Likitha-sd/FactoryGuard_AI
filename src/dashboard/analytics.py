from sqlalchemy.orm import Session

from src.database.crud import get_prediction_history


def load_dashboard_data(db: Session):

    history = get_prediction_history(db)

    total = len(history)

    normal = sum(
        1 for row in history
        if row.prediction == "Normal"
    )

    failure = total - normal

    return {
        "history": history,
        "total": total,
        "normal": normal,
        "failure": failure,
    }