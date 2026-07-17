from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# ==========================================================
# Database Location
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATABASE_PATH = BASE_DIR / "database" / "factoryguard.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# ==========================================================
# SQLAlchemy Engine
# ==========================================================

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# ==========================================================
# Dependency
# ==========================================================

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()