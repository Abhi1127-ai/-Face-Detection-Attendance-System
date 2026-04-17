from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.database.models import Teacher
from backend.services.auth_service import decode_access_token

# ─────────────────────────────────────────
# OAuth2 Scheme
# ─────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ─────────────────────────────────────────
# Current Teacher Get Karo
# ─────────────────────────────────────────
def get_current_teacher(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Teacher:
    """
    JWT token se current logged-in teacher nikalo
    Agar token invalid ho toh 401 error do
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token!",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)

    if payload is None:
        raise credentials_exception

    teacher_id: str = payload.get("sub")

    if teacher_id is None:
        raise credentials_exception

    teacher = db.query(Teacher).filter(Teacher.id == teacher_id).first()

    if teacher is None:
        raise credentials_exception

    return teacher