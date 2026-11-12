
import cv2
from app.camera import USBCamera


def main():
    camera = USBCamera(camera_index=0)

    try:
        camera.open()

        while True:
            success, frame = camera.read()

            if not success:
                print("Erro ao capturar imagem.")
                break

            cv2.imshow("Reali - Camera USB", frame)

            # ESC para encerrar
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except RuntimeError as error:
        print(error)

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
