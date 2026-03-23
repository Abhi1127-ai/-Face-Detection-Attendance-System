import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from backend.services.enrollment_service import load_embeddings
from backend.config import SIMILARITY_THRESHOLD

# Initialize AI Model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))


def recognize_faces_in_frame(frame):
    """
    Frame mein faces dhoondo aur known embeddings se match karo
    """
    known_ids, known_encodings = load_embeddings()
    if not known_ids:
        return []

    faces = app.get(frame)
    found_student_ids = []

    for face in faces:
        current_embedding = face.embedding

        # Sabse zyada match hone wala student dhoondo
        similarities = [1 - cosine(current_embedding, known) for known in known_encodings]
        max_sim = max(similarities) if similarities else 0

        if max_sim > SIMILARITY_THRESHOLD:
            best_match_idx = similarities.index(max_sim)
            found_student_ids.append(known_ids[best_match_idx])

    return found_student_ids