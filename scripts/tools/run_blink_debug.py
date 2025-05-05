# run_blink_debug.py

import cv2
from gaze.face_mesh import FaceMeshDetector
from gaze.blink_detector import BlinkDetector

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")
    
    # 1) Init les deux détecteurs
    fm = FaceMeshDetector(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    blink_detector = BlinkDetector()

    print("Clignements détectés: 0", end="", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        lm = fm.process(frame)
        if lm:
            blink_detector.update(frame,lm,fm)  
        count = blink_detector.blink_count

        # Affiche le compteur en haut à droite
        cv2.putText(
            frame,
            f"Blink: {count}",
            (frame.shape[1] - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Blink Debug", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print()  # saut de ligne après sortie
    
if __name__ == "__main__":
    main()
