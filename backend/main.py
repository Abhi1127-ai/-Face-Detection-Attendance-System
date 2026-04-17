from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.database.db import init_db
from backend.services.recognition_service import load_embeddings_to_memory
from backend.routes.auth_routes import router as auth_router
from backend.routes.enrollment_routes import router as enrollment_router
from backend.routes.recognition_routes import router as recognition_router
from backend.routes.teacher_route import router as teacher_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_embeddings_to_memory()
    print("[APP] Server started successfully!")
    yield


app = FastAPI(
    title="AI Face Recognition Attendance System",
    description="70 students ke liye AI based attendance system",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes first ──
app.include_router(auth_router)
app.include_router(enrollment_router)
app.include_router(recognition_router)
app.include_router(teacher_router)

# ── Health check ──
@app.get("/health")
async def root():
    return {"message": "Attendance System API is running! 🚀"}

# ── Static files LAST — warna "/" saari routes swallow kar lega ──
app.mount("/", StaticFiles(directory="backend/frontend", html=True), name="frontend")