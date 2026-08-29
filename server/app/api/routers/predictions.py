from fastapi import APIRouter

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


@router.get("/")
def predictions_placeholder():
    return {"status": "ok", "message": "Predictions endpoint"}
