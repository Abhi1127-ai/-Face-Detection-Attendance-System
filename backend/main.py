from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database.db import init_db




# ─────────────────────────────────────────
# Lifespan — DB initialize karo
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("[APP] Server started successfully!")
    yield


# ─────────────────────────────────────────
# App Initialize
# ─────────────────────────────────────────
app = FastAPI(
    title="AI Face Recognition Attendance System",
    description="70 students ke liye AI based attendance system",
    version="1.0.0",
    lifespan=lifespan
)



# ─────────────────────────────────────────
# CORS — Frontend se requests allow karo
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.routes.auth_routes import router as auth_router
app.include_router(auth_router)

from backend.routes.enrollment_routes import router as enrollment_router
app.include_router(enrollment_router)

# ─────────────────────────────────────────
# Test Route
# ─────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Attendance System API is running! 🚀"}