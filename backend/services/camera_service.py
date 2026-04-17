# backend/services/camera_service.py

class CameraService:
    def __init__(self):
        self.is_running = False

    def start(self):
        self.is_running = True
        print("[CAMERA] Browser mode — camera browser se handle hoga.")

    def stop(self):
        self.is_running = False
        print("[CAMERA] Stopped.")

    def is_camera_available(self):
        return True

    def capture_frame(self):
        raise Exception("Browser camera use ho rahi hai. Is function ki zaroorat nahi.")


# Global Instance
camera_service = CameraService()