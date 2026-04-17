from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
import enum

from backend.database.db import Base


# ─────────────────────────────────────────
# Enums
# ─────────────────────────────────────────

class AttendanceStatus(str, enum.Enum):
    PRESENT   = "PRESENT"
    ABSENT    = "ABSENT"
    UNCERTAIN = "UNCERTAIN"


class ScanType(str, enum.Enum):
    ENTRY  = "ENTRY"
    MID    = "MID"
    EXIT   = "EXIT"
    MANUAL = "MANUAL"


class ClassStatus(str, enum.Enum):
    ONGOING   = "ONGOING"
    COMPLETED = "COMPLETED"


# ─────────────────────────────────────────
# Teachers Table
# ─────────────────────────────────────────

class Teacher(Base):
    __tablename__ = "teachers"

    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name       = Column(String, nullable=False)
    email      = Column(String, unique=True, nullable=False)
    password   = Column(String, nullable=False)  # hashed password
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relations
    classes   = relationship("Class", back_populates="teacher")
    overrides = relationship("AttendanceOverride", back_populates="teacher")


# ─────────────────────────────────────────
# Students Table
# ─────────────────────────────────────────

class Student(Base):
    __tablename__ = "students"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name        = Column(String, nullable=False)
    roll_no     = Column(String, unique=True, nullable=False)
    photo_path  = Column(String, nullable=True)
    enrolled_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relations
    attendances = relationship("Attendance", back_populates="student")


# ─────────────────────────────────────────
# Classes Table
# ─────────────────────────────────────────

class Class(Base):
    __tablename__ = "classes"

    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    subject          = Column(String, nullable=False)
    teacher_id       = Column(String, ForeignKey("teachers.id"), nullable=False)
    date             = Column(String, nullable=False)
    start_time       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_time         = Column(DateTime, nullable=True)
    duration_minutes = Column(Integer, default=60)
    status           = Column(String, default=ClassStatus.ONGOING)

    # Relations
    teacher     = relationship("Teacher", back_populates="classes")
    attendances = relationship("Attendance", back_populates="class_")
    scan_logs   = relationship("ScanLog", back_populates="class_")


# ─────────────────────────────────────────
# Attendance Table
# ─────────────────────────────────────────

class Attendance(Base):
    __tablename__ = "attendance"

    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id       = Column(String, ForeignKey("students.id"), nullable=False)
    class_id         = Column(String, ForeignKey("classes.id"), nullable=False)
    status           = Column(String, default=AttendanceStatus.UNCERTAIN)
    confidence_score = Column(Float, nullable=True)
    strike_count     = Column(Integer, default=0)
    final_marked_at  = Column(DateTime, nullable=True)

    # Relations
    student   = relationship("Student", back_populates="attendances")
    class_    = relationship("Class", back_populates="attendances")
    overrides = relationship("AttendanceOverride", back_populates="attendance")


# ─────────────────────────────────────────
# Scan Logs Table
# ─────────────────────────────────────────

class ScanLog(Base):
    __tablename__ = "scan_logs"

    id                = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    class_id          = Column(String, ForeignKey("classes.id"), nullable=False)
    scan_type         = Column(String, nullable=False)
    triggered_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    students_detected = Column(Integer, default=0)

    # Relations
    class_ = relationship("Class", back_populates="scan_logs")


# ─────────────────────────────────────────
# Attendance Overrides Table
# ─────────────────────────────────────────

class AttendanceOverride(Base):
    __tablename__ = "attendance_overrides"

    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    attendance_id = Column(String, ForeignKey("attendance.id"), nullable=False)
    teacher_id    = Column(String, ForeignKey("teachers.id"), nullable=False)
    old_status    = Column(String, nullable=False)
    new_status    = Column(String, nullable=False)
    reason        = Column(String, nullable=True)
    overridden_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relations
    attendance = relationship("Attendance", back_populates="overrides")
    teacher    = relationship("Teacher", back_populates="overrides")