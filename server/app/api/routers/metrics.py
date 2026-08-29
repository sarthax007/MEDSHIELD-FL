from fastapi import APIRouter

router = APIRouter(
    prefix="/metrics",
    tags=["Metrics"],
)


@router.get("/")
def metrics_placeholder():
    return {"status": "ok", "message": "Metrics endpoint"}
