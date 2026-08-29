from fastapi import APIRouter

router = APIRouter(
    prefix="/rounds",
    tags=["Training Rounds"],
)


@router.get("/")
def rounds_placeholder():
    return {"status": "ok", "message": "Training rounds endpoint"}
