from fastapi import APIRouter

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.get("/")
def auth_placeholder():
    return {"status": "ok", "message": "Auth endpoint"}
