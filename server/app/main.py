from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

from app.api.routers import (
    explain,
    auth,
    hospitals,
    rounds,
    predictions,
    labelling,
    metrics,
)
from app.db.session import get_db

app = FastAPI(title="MedShield-FL Backend")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": str(exc)},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "details": "HTTPException"},
    )


app.include_router(explain.router)
app.include_router(auth.router)
app.include_router(hospitals.router)
app.include_router(rounds.router)
app.include_router(predictions.router)
app.include_router(labelling.router)
app.include_router(metrics.router)


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {"status": "ok", "service": "backend", "database": db_status}
