from fastapi import APIRouter

router = APIRouter(
    prefix="/hospitals",
    tags=["Hospitals"],
)


@router.get("/")
def hospitals_placeholder():
    return {"status": "ok", "message": "Hospitals endpoint"}
