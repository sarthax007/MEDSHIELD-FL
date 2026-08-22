from fastapi import FastAPI

from app.api.routers import explain

app = FastAPI(title="MedShield-FL Backend")

app.include_router(explain.router)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "backend"}
