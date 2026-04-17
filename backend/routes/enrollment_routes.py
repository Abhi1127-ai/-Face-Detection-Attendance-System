from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import cv2
import numpy as np

from backend.database.db import get_db
from backend.middleware.auth_middleware import get_current_teacher
from backend.services.enrollment_service import (
    enroll_student_via_upload,
    enroll_student_via_camera,
    get_all_students,
    delete_student
)
from pydantic import BaseModel

router = APIRouter(
    prefix="/enrollment",
    tags=["Enrollment"]
)


# ─────────────────────────────────────────
# Pydantic Schema
# ─────────────────────────────────────────
class StudentResponse(BaseModel):
    id: str
    name: str
    roll_no: str
    photo_path: str | None

    class Config:
        from_attributes = True


# ─────────────────────────────────────────
# Get All Students
# ─────────────────────────────────────────
@router.get("/students", response_model=List[StudentResponse])
async def get_students(
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """Saare enrolled students ki list"""
    return get_all_students(db)


# ─────────────────────────────────────────
# Option 1 — Image Upload Se Enroll
# ─────────────────────────────────────────
@router.post("/student/upload", status_code=status.HTTP_201_CREATED)
async def enroll_via_upload(
    name: str,
    roll_no: str,
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """
    Image upload se student enroll karo
    3-5 photos upload karo alag alag angles se
    """
    np_images = []
    for image in images:
        file_bytes = await image.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            np_images.append(frame)

    if not np_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Koi valid image nahi mili!"
        )

    try:
        student = enroll_student_via_upload(
            db=db,
            name=name,
            roll_no=roll_no,
            images=np_images
        )
        return {
            "message": f"Student {name} successfully enrolled via upload!",
            "student_id": student.id,
            "roll_no": student.roll_no
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ─────────────────────────────────────────
# Option 2 — Camera Se Enroll
# ─────────────────────────────────────────
@router.post("/student/camera", status_code=status.HTTP_201_CREATED)
async def enroll_via_camera(
    name: str,
    roll_no: str,
    num_photos: int = 3,
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """
    Camera se student enroll karo
    - Camera window khulegi
    - 'S' dabao photo lene ke liye
    - 'Q' dabao quit karne ke liye
    """
    try:
        student = enroll_student_via_camera(
            db=db,
            name=name,
            roll_no=roll_no,
            num_photos=num_photos
        )
        return {
            "message": f"Student {name} successfully enrolled via camera!",
            "student_id": student.id,
            "roll_no": student.roll_no
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ─────────────────────────────────────────
# Delete Student
# ─────────────────────────────────────────
@router.delete("/student/{student_id}")
async def remove_student(
    student_id: str,
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """Delete the student from the system"""
    success = delete_student(db, student_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student nahi mila!"
        )

    return {"message": "Student successfully removed!"}