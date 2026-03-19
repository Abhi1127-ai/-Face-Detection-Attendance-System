import os
import pickle
import numpy as np
from sqlalchemy.orm import Session
from backend.database.models import Student
from backend.config import STUDENT_FACES_DIR, EMBEDDINGS_PATH

# ─────────────────────────────────────────
# Embeddings Load/Save
# ─────────────────────────────────────────

def load_embeddings() -> tuple:
    """Pickle file se embeddings load karo"""
    if not os.path.exists(EMBEDDINGS_PATH):
        return [], []
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["ids"], data["encodings"]


def save_embeddings(known_ids: list, known_encodings: list):
    """Embeddings pickle file mein save karo"""
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({
            "ids": known_ids,
            "encodings": known_encodings
        }, f)
    print(f"[EMBEDDINGS] Saved {len(known_ids)} students!")


# ─────────────────────────────────────────
# Student Enrollment
# ─────────────────────────────────────────

def enroll_student(
    db: Session,
    name: str,
    roll_no: str,
    images: list
) -> Student:
    """
    Student ko enroll karo:
    1. DB mein save karo
    2. Face embeddings nikalo
    3. Average embedding pickle mein save karo
    """
    from insightface.app import FaceAnalysis

    # InsightFace app initialize karo
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Embeddings nikalo saari images se
    embeddings = []
    for img in images:
        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)

    if not embeddings:
        raise ValueError("Kisi bhi image mein face detect nahi hua!")

    # Average embedding nikalo
    avg_embedding = np.mean(embeddings, axis=0)

    # Student folder banao agar exist nahi karta
    os.makedirs(STUDENT_FACES_DIR, exist_ok=True)

    # DB mein student save karo
    student = Student(
        name=name,
        roll_no=roll_no,
        photo_path=f"{STUDENT_FACES_DIR}/{roll_no}.jpg"
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    # Existing embeddings load karo aur naya add karo
    known_ids, known_encodings = load_embeddings()
    known_ids.append(student.id)
    known_encodings.append(avg_embedding)
    save_embeddings(known_ids, known_encodings)

    print(f"[ENROLLMENT] Student {name} ({roll_no}) enrolled successfully!")
    return student


def get_all_students(db: Session) -> list:
    """Saare enrolled students return karo"""
    return db.query(Student).all()


def delete_student(db: Session, student_id: str) -> bool:
    """
    Student ko delete karo:
    1. DB se remove karo
    2. Embedding bhi remove karo
    """
    student = db.query(Student).filter(Student.id == student_id).first()

    if not student:
        return False

    # Embedding remove karo
    known_ids, known_encodings = load_embeddings()
    if student.id in known_ids:
        idx = known_ids.index(student.id)
        known_ids.pop(idx)
        known_encodings.pop(idx)
        save_embeddings(known_ids, known_encodings)

    # DB se delete karo
    db.delete(student)
    db.commit()

    print(f"[ENROLLMENT] Student {student.name} deleted!")
    return True