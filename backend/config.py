import os

# ─────────────────────────────────────────
# Face Recognition Settings
# ─────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.35  # is se kam → Unknown
CONFIRMED_THRESHOLD  = 0.55   # is se zyada → Confirmed Present
ENROLLMENT_PHOTOS    = 5   # enrollment mein kitni photos leni hain

# ─────────────────────────────────────────
# Attendance Logic Settings
# ─────────────────────────────────────────
STRIKE_LIMIT             = 1  # kitne strikes pe absent mark karo
ENTRY_WINDOW_MINUTES     = 1  # class shuru ke baad kitne min tak entry scan
EXIT_WINDOW_MINUTES      = 1  # class khatam hone se pehle kitne min
MID_SCAN_COUNT_MIN       = 1  # minimum random mid scans
MID_SCAN_COUNT_MAX       = 2  # maximum random mid scans
ENTRY_SCAN_INTERVAL_SECONDS = 10  # Har 10 sec pe entry scan
EXIT_SCAN_INTERVAL_SECONDS = 10   # Har 10 sec pe exit scan

# ─────────────────────────────────────────
# Camera Settings
# ─────────────────────────────────────────
CAMERA_INDEX = 0               # USB webcam index (0 = default)

# ─────────────────────────────────────────
# JWT Auth Settings
# ─────────────────────────────────────────
SECRET_KEY       = "change-this-to-a-strong-secret-key"
ALGORITHM        = "HS256"
TOKEN_EXPIRE_MIN = 60 * 8      # 8 hours

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
STUDENT_FACES_DIR = "student_faces"
EMBEDDINGS_PATH   = "embeddings.pkl"
DATABASE_URL      = "sqlite:///./attendance.db"