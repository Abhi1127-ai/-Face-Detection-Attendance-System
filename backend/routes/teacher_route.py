import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import Class, Attendance, Student
from backend.middleware.auth_middleware import get_current_teacher
from backend.services.attendance_service import (
    start_class_session,
    end_class_session,
    get_attendance_report,
    override_attendance
)
from backend.services.camera_service import camera_service
from backend.services.scheduler_service import run_class_session, stop_session
from pydantic import BaseModel

router = APIRouter(
    prefix="/teacher",
    tags=["Teacher"]
)


# ─────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────

class StartClassRequest(BaseModel):
    subject: str
    duration_minutes: int = 60


class OverrideRequest(BaseModel):
    attendance_id: str
    new_status: str
    reason: str = None


# ─────────────────────────────────────────
# Start Class
# ─────────────────────────────────────────

@router.post("/class/start")
async def start_class(
    request: StartClassRequest,
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """
    Class shuru karo:
    1. Camera on karo
    2. DB mein class create karo
    3. Automatic scanning shuru karo
    """
    # Camera available hai?
    if not camera_service.is_camera_available():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera nahi mila! Connection check karo."
        )

    # Camera start karo
    camera_service.start()

    # DB mein class create karo
    new_class = start_class_session(
        db=db,
        subject=request.subject,
        teacher_id=teacher.id,
        duration_minutes=request.duration_minutes
    )

    # Background mein scheduler start karo
    asyncio.create_task(
        run_class_session(
            class_id=new_class.id,
            duration_minutes=request.duration_minutes
        )
    )

    return {
        "message": f"Class '{request.subject}' started!",
        "class_id": new_class.id,
        "duration_minutes": request.duration_minutes
    }


# ─────────────────────────────────────────
# End Class
# ─────────────────────────────────────────

@router.post("/class/end/{class_id}")
async def end_class(
    class_id: str,
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """
    Class khatam karo:
    1. Scheduler stop karo
    2. Camera off karo
    3. Final attendance mark karo
    4. Report return karo
    """
    # Scheduler stop karo
    stop_session()

    # Camera stop karo
    camera_service.stop()

    # Final attendance mark karo
    report = end_class_session(db=db, class_id=class_id)

    return {
        "message": "Class ended!",
        "report": report
    }


# ─────────────────────────────────────────
# Manual Scan
# ─────────────────────────────────────────

@router.post("/scan/manual/{class_id}")
async def manual_scan(
    class_id: str,
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """Teacher manually scan trigger kare"""
    from backend.services.scheduler_service import trigger_scan
    await trigger_scan(class_id, "MANUAL")
    return {"message": "Manual scan triggered!"}


# ─────────────────────────────────────────
# Attendance Report
# ─────────────────────────────────────────

@router.get("/attendance/{class_id}")
async def get_report(
    class_id: str,
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """Class ki attendance report lo"""
    report = get_attendance_report(db=db, class_id=class_id)

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class nahi mili ya koi student nahi!"
        )

    return {
        "class_id": class_id,
        "total_students": len(report),
        "report": report
    }


# ─────────────────────────────────────────
# Override Attendance
# ─────────────────────────────────────────

@router.post("/attendance/override")
async def override(
    request: OverrideRequest,
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """Teacher manually attendance change kare"""
    success = override_attendance(
        db=db,
        attendance_id=request.attendance_id,
        teacher_id=teacher.id,
        new_status=request.new_status,
        reason=request.reason
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Attendance record nahi mila!"
        )

    return {"message": "Attendance successfully overridden!"}


# ─────────────────────────────────────────
# All Classes
# ─────────────────────────────────────────

@router.get("/classes")
async def get_classes(
    db: Session = Depends(get_db),
    teacher=Depends(get_current_teacher)
):
    """Teacher ki saari classes lo"""
    classes = db.query(Class).filter(
        Class.teacher_id == teacher.id
    ).all()

    return {
        "total": len(classes),
        "classes": [
            {
                "id": c.id,
                "subject": c.subject,
                "date": c.date,
                "status": c.status,
                "duration_minutes": c.duration_minutes
            }
            for c in classes
        ]
    }