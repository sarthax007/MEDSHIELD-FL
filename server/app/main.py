from fastapi import FastAPI

app = FastAPI(title="MedShield-FL Backend")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "backend"}
