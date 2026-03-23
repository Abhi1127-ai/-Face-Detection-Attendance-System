from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from typing import List
import cv2
import numpy as np
from datetime import datetime

from backend.database.db import get_db
from backend.database.models import Student, Attendance, Class, ScanLog
from backend.services.attendance_service import recognize_faces_in_frame
from backend.middleware.auth_middleware import get_current_teacher

router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.post("/recognize")
async def recognize(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        teacher=Depends(get_current_teacher)
):
    """
    Frontend se image aayegi, AI check karega, aur DB mein 'Present' mark karega
    """
    # Image process karo
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Face Recognition
    matched_ids = recognize_faces_in_frame(frame)

    present_names = []
    for s_id in matched_ids:
        student = db.query(Student).filter(Student.id == s_id).first()
        if student:
            present_names.append(student.name)
            # Yahan aap Attendance table mein entry update kar sakte hain
            # (e.g. status = "PRESENT", final_marked_at = now)

    return {"present_students": present_names}


@router.get("/export")
async def export_attendance(date: str, db: Session = Depends(get_db), teacher=Depends(get_current_teacher)):
    """
    CSV Generator logic yahan aayegi (as requested for Admin panel)
    """
    # Logic to query DB for 'date' and return StreamingResponse(csv_file)
    return {"message": f"Exporting data for {date}"}