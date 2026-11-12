
import time
from app.camera import USBCamera

def main():
    camera = USBCamera(
        camera_index=0,
        width=1280,
        height=720,
        fps=30
    )

    camera.start()

    print("Testando captura em thread por 10 segundos...")

    try:
        time.sleep(10)
    finally:
        camera.stop()

if __name__ == "__main__":
    main()
