from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from app.core import security
from app.core.config import settings
from app.db.models import User as DBUser
from app.schemas.token import Token, TokenPayload
from app.schemas.user import User
from app.api.deps import get_db, get_current_active_user

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

@router.post("/login", response_model=Token)
def login_access_token(
    db: Annotated[Session, Depends(get_db)],
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = db.query(DBUser).filter(DBUser.username == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        subject=user.username, expires_delta=access_token_expires
    )
    refresh_token = security.create_refresh_token(subject=user.username)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


@router.post("/refresh", response_model=Token)
def refresh_access_token(
    db: Annotated[Session, Depends(get_db)],
    refresh_token: str,
):
    """
    Refresh an access token using a valid refresh token.
    """
    try:
        payload = jwt.decode(
            refresh_token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        if token_data.type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials: wrong token type",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )

    user = db.query(DBUser).filter(DBUser.username == token_data.sub).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        subject=user.username, expires_delta=access_token_expires
    )
    new_refresh_token = security.create_refresh_token(subject=user.username)

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
    )


@router.get("/me", response_model=User)
def read_current_user(
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
):
    """
    Get current user.
    """
    return current_user
