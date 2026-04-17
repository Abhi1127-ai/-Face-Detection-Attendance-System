import os
import pickle
import cv2
import numpy as np
from sqlalchemy.orm import Session
from backend.database.models import Student
from backend.config import STUDENT_FACES_DIR, EMBEDDINGS_PATH


# ─────────────────────────────────────────
# Embeddings Load/Save
# ─────────────────────────────────────────

def load_embeddings() -> tuple:
    if not os.path.exists(EMBEDDINGS_PATH):
        return [], []
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["ids"], data["encodings"]


def save_embeddings(known_ids: list, known_encodings: list):
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({
            "ids": known_ids,
            "encodings": known_encodings
        }, f)
    print(f"[EMBEDDINGS] Saved {len(known_ids)} students!")


# ─────────────────────────────────────────
# Image Quality Check
# ─────────────────────────────────────────

def is_image_quality_good(image: np.ndarray) -> bool:
    """Blur check karo"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 50:
        print(f"[QUALITY] Image too blurry! Score: {blur_score}")
        return False
    return True


# ─────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """CLAHE se lighting enhance karo"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return enhanced


# ─────────────────────────────────────────
# Core Enrollment Logic
# ─────────────────────────────────────────

def process_enrollment(
    db: Session,
    name: str,
    roll_no: str,
    images: list
) -> Student:
    """Core enrollment — dono options yahi use karenge"""
    
    from backend.services.recognition_service import app

    embeddings = []
    rejected = 0

    for img in images:
        if not is_image_quality_good(img):
            rejected += 1
            continue

        img = preprocess_image(img)
        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)

    if not embeddings:
        raise ValueError(
            f"Koi valid face nahi mila! "
            f"{rejected} images quality check mein fail hui."
        )

    # Average embedding
    avg_embedding = np.mean(embeddings, axis=0)

    # Student folder
    os.makedirs(STUDENT_FACES_DIR, exist_ok=True)

    # DB mein save
    student = Student(
        name=name,
        roll_no=roll_no,
        photo_path=f"{STUDENT_FACES_DIR}/{roll_no}.jpg"
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    # Embeddings update
    known_ids, known_encodings = load_embeddings()
    known_ids.append(student.id)
    known_encodings.append(avg_embedding)
    save_embeddings(known_ids, known_encodings)

    print(f"[ENROLLMENT] {name} ({roll_no}) enrolled! "
          f"({len(embeddings)} images used, {rejected} rejected)")
    return student


# ─────────────────────────────────────────
# Option 1 — Image Upload Se Enroll
# ─────────────────────────────────────────

def enroll_student_via_upload(
    db: Session,
    name: str,
    roll_no: str,
    images: list
) -> Student:
    """Image upload se student enroll karo"""
    return process_enrollment(db, name, roll_no, images)


# ─────────────────────────────────────────
# Option 2 — Camera Se Enroll
# ─────────────────────────────────────────

def enroll_student_via_camera(
    db: Session,
    name: str,
    roll_no: str,
    num_photos: int = 5
) -> Student:
    """
    Camera se 3 photos lo:
    Centre → Left → Right
    """
    instructions = [
        "Look straight",
        "Turn your face left",
        "Turn your face right",
        "Tilt your head up",
        "Tilt your head down"
    ]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Can't open the comera! Ensure it's connected and accessible.")

    images = []
    print("\n[CAMERA] Enrollment can be done now!")
    print("[CAMERA] Press 's' to capture photos and to quit press 'q'\n")

    while len(images) < num_photos:
        ret, frame = cap.read()
        if not ret:
            break

        # Current instruction
        instruction = instructions[len(images)]

        cv2.putText(
            frame,
            f"{instruction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"Photo {len(images)+1}/3",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 0), 2
        )
        cv2.putText(
            frame,
            "'S' = Photo Lo | 'Q' = Quit",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        cv2.imshow(f"Enrollment - {name}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            images.append(frame.copy())
            print(f"[CAMERA] Photo {len(images)}/3 li — {instruction}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not images:
        raise ValueError("No photo has been captured!")

    return process_enrollment(db, name, roll_no, images)


# ─────────────────────────────────────────
# Get All Students
# ─────────────────────────────────────────

def get_all_students(db: Session) -> list:
    return db.query(Student).all()


# ─────────────────────────────────────────
# Delete Student
# ─────────────────────────────────────────
def delete_student(db: Session, student_id: str) -> bool:
    student = db.query(Student).filter(Student.id == student_id).first()

    if not student:
        return False

    # ✅ Pehle attendance records delete karo
    from backend.database.models import Attendance
    db.query(Attendance).filter(
        Attendance.student_id == student_id
    ).delete()

    # ✅ Embeddings se remove karo
    known_ids, known_encodings = load_embeddings()
    if student.id in known_ids:
        idx = known_ids.index(student.id)
        known_ids.pop(idx)
        known_encodings.pop(idx)
        save_embeddings(known_ids, known_encodings)

        from backend.services.recognition_service import reload_embeddings
        reload_embeddings()

    # ✅ Ab student delete karo
    db.delete(student)
    db.commit()

    print(f"[ENROLLMENT] Student {student.name} deleted!")
    return True