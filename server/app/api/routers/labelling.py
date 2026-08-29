from fastapi import APIRouter

router = APIRouter(
    prefix="/labelling",
    tags=["Labelling"],
)


@router.get("/")
def labelling_placeholder():
    return {"status": "ok", "message": "Labelling endpoint"}
