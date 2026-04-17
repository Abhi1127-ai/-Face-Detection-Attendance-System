import asyncio
import random
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.database.db import SessionLocal
from backend.config import (
    MID_SCAN_COUNT_MIN,
    MID_SCAN_COUNT_MAX,
    ENTRY_WINDOW_MINUTES,
    EXIT_WINDOW_MINUTES,
    ENTRY_SCAN_INTERVAL_SECONDS,
    EXIT_SCAN_INTERVAL_SECONDS
)


# ─────────────────────────────────────────
# Global State
# ─────────────────────────────────────────
current_class_id = None
is_session_active = False


# ─────────────────────────────────────────
# Trigger Scan
# ─────────────────────────────────────────

async def trigger_scan(class_id: str, scan_type: str):
    from backend.services.camera_service import camera_service
    from backend.services.recognition_service import recognize_all_faces
    from backend.services.attendance_service import process_scan_results
    from backend.database.models import ScanLog

    print(f"\n[SCAN] {scan_type} scan triggered at {datetime.now(timezone.utc)}")

    db: Session = SessionLocal()

    try:
        # ✅ 5 frames lo, best results merge karo
        all_detected = {}  # {student_id: best_confidence}
        
        for i in range(5):
            frame = camera_service.capture_frame()
            detected = recognize_all_faces(frame)
            
            for student in detected:
                sid = student["student_id"]
                if sid is None:
                    continue
                # Best confidence track karo
                if sid not in all_detected or student["confidence"] > all_detected[sid]["confidence"]:
                    all_detected[sid] = student
            
            await asyncio.sleep(0.4)  # 0.4 sec gap — 5 frames = 2 sec

        # Dict se list banao
        detected_students = list(all_detected.values())

        # Attendance update karo
        result = process_scan_results(
            db=db,
            class_id=class_id,
            scan_type=scan_type,
            detected_students=detected_students
        )

        # Scan log save karo
        scan_log = ScanLog(
            class_id=class_id,
            scan_type=scan_type,
            students_detected=len(detected_students)
        )
        db.add(scan_log)
        db.commit()

        print(f"[SCAN] {scan_type} complete! "
              f"Detected: {result['detected']}, "
              f"Alerts: {len(result['alerts'])}")

        for alert in result["alerts"]:
            print(f"[ALERT] {alert['student_name']} "
                  f"({alert['roll_no']}) — "
                  f"{alert['strikes']} strikes!")

    except Exception as e:
        print(f"[ERROR] Scan failed: {e}")

    finally:
        db.close()


# ─────────────────────────────────────────
# Entry Window — 5 min tak continuously scan
# ─────────────────────────────────────────

async def run_entry_window(class_id: str):
    """
    Pehle 5 min tak har X seconds pe scan karo
    Jaise hi student detect ho — present mark ho jaayega
    No strikes during entry window
    """
    print(f"[SCHEDULER] Entry window started! ({ENTRY_WINDOW_MINUTES} min)")

    entry_duration = ENTRY_WINDOW_MINUTES * 60  # seconds mein
    elapsed = 0

    while elapsed < entry_duration and is_session_active:
        await trigger_scan(class_id, "ENTRY")
        await asyncio.sleep(ENTRY_SCAN_INTERVAL_SECONDS)
        elapsed += ENTRY_SCAN_INTERVAL_SECONDS

    print(f"[SCHEDULER] Entry window complete!")


# ─────────────────────────────────────────
# Exit Window — last 5 min tak continuously scan
# ─────────────────────────────────────────

async def run_exit_window(class_id: str):
    """
    Last 5 min tak har X seconds pe scan karo
    Confirm karo kaun abhi bhi baitha hai
    """
    print(f"[SCHEDULER] Exit window started! ({EXIT_WINDOW_MINUTES} min)")

    exit_duration = EXIT_WINDOW_MINUTES * 60  # seconds mein
    elapsed = 0

    while elapsed < exit_duration and is_session_active:
        await trigger_scan(class_id, "EXIT")
        await asyncio.sleep(EXIT_SCAN_INTERVAL_SECONDS)
        elapsed += EXIT_SCAN_INTERVAL_SECONDS

    print(f"[SCHEDULER] Exit window complete!")


# ─────────────────────────────────────────
# Random Mid Scans
# ─────────────────────────────────────────

async def schedule_mid_scans(class_id: str, duration_minutes: int):
    """
    Class ke beech mein 2-3 random mid scans karo
    Strike system active rahega
    """
    num_scans = random.randint(MID_SCAN_COUNT_MIN, MID_SCAN_COUNT_MAX)

    available_start = ENTRY_WINDOW_MINUTES
    available_end = duration_minutes - EXIT_WINDOW_MINUTES

    scan_times = sorted(random.sample(
        range(available_start, available_end),
        num_scans
    ))

    print(f"[SCHEDULER] Mid scans scheduled at minutes: {scan_times}")

    last_time = ENTRY_WINDOW_MINUTES
    for scan_minute in scan_times:
        if not is_session_active:
            break

        wait_seconds = (scan_minute - last_time) * 60
        await asyncio.sleep(wait_seconds)

        if is_session_active:
            await trigger_scan(class_id, "MID")

        last_time = scan_minute


# ─────────────────────────────────────────
# Main Class Session
# ─────────────────────────────────────────

async def run_class_session(class_id: str, duration_minutes: int = 50):
    """
    Poora class session manage karo:
    Entry Window → Mid Scans → Exit Window
    """
    global current_class_id, is_session_active

    current_class_id = class_id
    is_session_active = True

    print(f"\n[SCHEDULER] Class session started! Duration: {duration_minutes} min")

    # Step 1 — Entry Window (pehle 5 min)
    await run_entry_window(class_id)

    # Step 2 — Mid Scans (random times pe)
    mid_scan_task = asyncio.create_task(
        schedule_mid_scans(class_id, duration_minutes)
    )

    # Step 3 — Exit Window ke liye wait karo
    mid_duration = (duration_minutes - ENTRY_WINDOW_MINUTES - EXIT_WINDOW_MINUTES) * 60
    await asyncio.sleep(mid_duration)

    # Mid scans cancel karo
    mid_scan_task.cancel()

    # Step 4 — Exit Window (last 5 min)
    if is_session_active:
        await run_exit_window(class_id)

    is_session_active = False
    print(f"[SCHEDULER] Class session complete!")


# ─────────────────────────────────────────
# Stop Session
# ─────────────────────────────────────────

def stop_session():
    """Session force stop karo"""
    global is_session_active
    is_session_active = False
    print("[SCHEDULER] Session stopped!")