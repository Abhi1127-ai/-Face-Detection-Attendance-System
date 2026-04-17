import pickle
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from backend.config import (
    EMBEDDINGS_PATH,
    SIMILARITY_THRESHOLD,
    CONFIRMED_THRESHOLD
)

# ─────────────────────────────────────────
# Global Model — CPU mode (ctx_id=-1)
# ctx_id=0  → GPU/device 0 → webcam conflict on laptops!
# ctx_id=-1 → CPU mode    → no hardware conflict
# ─────────────────────────────────────────
_face_app = None

def get_face_app():
    """Lazy load — model sirf pehli baar use pe load hoga"""
    global _face_app
    if _face_app is None:
        print("[RECOGNITION] Loading InsightFace model (CPU mode)...")
        _face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        _face_app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[RECOGNITION] Model loaded!")
    return _face_app


# ─────────────────────────────────────────
# Embeddings — Memory mein store
# ─────────────────────────────────────────
known_ids       = []
known_encodings = []


def load_embeddings_to_memory():
    """Server start hote hi embeddings memory mein load karo"""
    global known_ids, known_encodings
    try:
        with open(EMBEDDINGS_PATH, "rb") as f:
            data = pickle.load(f)
        known_ids       = data.get("ids",       [])
        known_encodings = data.get("encodings", [])
        print(f"[RECOGNITION] {len(known_ids)} student embeddings loaded!")
    except FileNotFoundError:
        known_ids       = []
        known_encodings = []
        print("[RECOGNITION] No embeddings file found — enroll students first!")


def reload_embeddings():
    """Naya student enroll hone ke baad call karo"""
    load_embeddings_to_memory()


# ─────────────────────────────────────────
# Liveness Detection
# ─────────────────────────────────────────
def is_real_face(frame: np.ndarray, bbox) -> bool:
    """
    Basic liveness — texture sharpness check.
    Classroom lighting ke liye optimized.
    Threshold 25 — adjust if getting false SPOOF results.
    """
    x1, y1, x2, y2 = [int(b) for b in bbox]

    # Bounds check
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_region = frame[y1:y2, x1:x2]
    if face_region.size == 0:
        return False

    gray          = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if texture_score < 25:
        print(f"[LIVENESS] Failed — texture score: {texture_score:.1f}")
        return False

    print(f"[LIVENESS] Passed — texture score: {texture_score:.1f}")
    return True


# ─────────────────────────────────────────
# Single Face Recognize
# ─────────────────────────────────────────
def recognize_face(frame: np.ndarray) -> tuple:
    """
    Ek frame mein single best face recognize karo.
    Returns: (student_id, result_message, confidence_score)
    """
    face_app = get_face_app()
    faces    = face_app.get(frame)

    if not faces:
        return None, "No face detected", 0.0

    if not known_encodings:
        return None, "No students enrolled yet!", 0.0

    # Largest face use karo (closest to camera)
    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
    face  = faces[0]

    # Liveness check
    if not is_real_face(frame, face.bbox):
        return None, "Liveness check failed — real face required.", 0.0

    query   = face.embedding.reshape(1, -1)
    matrix  = np.array(known_encodings)
    scores  = cosine_similarity(query, matrix)[0]

    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= CONFIRMED_THRESHOLD:
        return known_ids[best_idx], f"CONFIRMED ({best_score:.2f})", round(best_score, 2)
    elif best_score >= SIMILARITY_THRESHOLD:
        return known_ids[best_idx], f"UNCERTAIN ({best_score:.2f})", round(best_score, 2)
    else:
        return None, f"UNKNOWN ({best_score:.2f})", round(best_score, 2)


# ─────────────────────────────────────────
# All Faces — Full Classroom Scan
# ─────────────────────────────────────────
def recognize_all_faces(frame: np.ndarray) -> list:
    """
    Ek frame mein saare faces recognize karo.
    Returns list of dicts with student_id, status, confidence, bbox.
    """
    face_app = get_face_app()
    faces    = face_app.get(frame)

    if not faces:
        return []

    if not known_encodings:
        # No students enrolled — return all as UNKNOWN with bbox
        return [{
            "student_id": None,
            "status":     "UNKNOWN",
            "confidence": 0.0,
            "bbox":       face.bbox.tolist()
        } for face in faces]

    query_embeddings  = np.array([f.embedding for f in faces])
    known_matrix      = np.array(known_encodings)
    similarity_matrix = cosine_similarity(query_embeddings, known_matrix)

    results = []
    for i, scores in enumerate(similarity_matrix):
        face       = faces[i]
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        # Liveness check first
        if not is_real_face(frame, face.bbox):
            results.append({
                "student_id": None,
                "status":     "SPOOF",
                "confidence": 0.0,
                "bbox":       face.bbox.tolist()
            })
            continue

        if best_score >= CONFIRMED_THRESHOLD:
            results.append({
                "student_id": known_ids[best_idx],
                "status":     "CONFIRMED",
                "confidence": round(best_score, 2),
                "bbox":       face.bbox.tolist()
            })
        elif best_score >= SIMILARITY_THRESHOLD:
            results.append({
                "student_id": known_ids[best_idx],
                "status":     "UNCERTAIN",
                "confidence": round(best_score, 2),
                "bbox":       face.bbox.tolist()
            })
        else:
            results.append({
                "student_id": None,
                "status":     "UNKNOWN",
                "confidence": round(best_score, 2),
                "bbox":       face.bbox.tolist()
            })

    return results