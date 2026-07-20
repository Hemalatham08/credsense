import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# Set this in a .env file next to this project, e.g.:
# DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/cardiosense
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/cardiosense"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# FastAPI dependency - gives each request its own DB session, closes it after
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()