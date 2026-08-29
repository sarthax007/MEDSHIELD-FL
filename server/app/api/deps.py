from typing import Generator, Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core import security
from app.core.config import settings
from app.db.session import get_db
from app.db.models import User
from app.schemas.token import TokenPayload

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"/auth/login"
)

SessionDep = Annotated[Session, Depends(get_db)]
TokenDep = Annotated[str, Depends(reusable_oauth2)]


def get_current_user(
    db: SessionDep, token: TokenDep
) -> User:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        if token_data.type != "access":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials: wrong token type",
            )
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = db.query(User).filter(User.username == token_data.sub).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    return current_user


def get_current_admin_user(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user


def get_current_doctor_user(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user


def get_current_operator_user(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    if current_user.role != "operator":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user
