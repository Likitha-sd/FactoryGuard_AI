from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import Float
from sqlalchemy import String
from sqlalchemy import DateTime

from datetime import datetime

from .database import Base


class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    machine_id = Column(String)

    temperature = Column(Float)

    vibration = Column(Float)

    pressure = Column(Float)

    rpm = Column(Float)

    power = Column(Float)

    operating_hours = Column(Float)

    prediction = Column(String)

    probability = Column(Float)