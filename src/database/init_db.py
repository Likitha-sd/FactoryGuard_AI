from .database import Base, engine

# Import all models so SQLAlchemy knows about them
from . import models


def initialize_database():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    initialize_database()
    print("✅ Database initialized successfully.")