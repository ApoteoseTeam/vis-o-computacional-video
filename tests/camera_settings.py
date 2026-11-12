
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a webcam")

try:
    print("Abrindo configurações da câmera...")

    resultado = cap.set(
        cv2.CAP_PROP_SETTINGS,
        1
    )

    print("Resultado:", resultado)
    print("Pressione ENTER após fechar as configurações.")
    input()

finally:
    cap.release()
