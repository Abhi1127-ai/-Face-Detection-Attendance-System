from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import cv2
import numpy as np

from backend.database.db import get_db
from backend.database.models import Student
from backend.middleware.auth_middleware import get_current_teacher
from backend.services.recognition_service import recognize_face, recognize_all_faces

router = APIRouter(
    prefix="/recognition",
    tags=["Recognition"]
)


# ─────────────────────────────────────────
# Single Face Recognize
# ─────────────────────────────────────────
@router.post("/recognize")
async def recognize(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """
    Ek image mein face recognize karo
    Student ka naam + status return karega
    """
    file_bytes = await image.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image!")

    student_id, result, confidence = recognize_face(frame)

    # Student ka naam DB se lo
    student_name = None
    if student_id:
        student = db.query(Student).filter(Student.id == student_id).first()
        if student:
            student_name = student.name

    return {
        "student_id": student_id,
        "student_name": student_name,
        "result": result,
        "confidence": confidence
    }


# ─────────────────────────────────────────
# Full Classroom Scan
# ─────────────────────────────────────────
@router.post("/scan")
async def scan_classroom(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """
    Ek frame mein saare faces recognize karo
    Classroom attendance scan ke liye
    """
    file_bytes = await image.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image!")

    detected = recognize_all_faces(frame)

    # Student names DB se lo
    results = []
    for d in detected:
        student_name = None
        if d["student_id"]:
            student = db.query(Student).filter(
                Student.id == d["student_id"]
            ).first()
            if student:
                student_name = student.name

        results.append({
            "student_id": d["student_id"],
            "student_name": student_name,
            "status": d["status"],
            "confidence": d["confidence"],
            "bbox": d["bbox"]
        })

    return {
        "total_detected": len(detected),
        "results": results
    }