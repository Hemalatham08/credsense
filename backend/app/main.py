from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .patients.routes import router as patients_router
from .predictions.routes import router as predictions_router
from .reports.routes import router as reports_router
from .analytics.routes import router as analytics_router
# Creates any tables defined in models.py that don't already exist.
# Since you ran schema.sql manually, this is mostly a safety net.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="CardioSense API", version="0.1.0")

# Allow the React frontend (running on a different port) to call this API.
# No auth/JWT middleware yet - login is a future enhancement.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients_router)
app.include_router(predictions_router)
app.include_router(reports_router)
app.include_router(analytics_router)

@app.get("/")
def health_check():
    return {"status": "CardioSense API is running"}