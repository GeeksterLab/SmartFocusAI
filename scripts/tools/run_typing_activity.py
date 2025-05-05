# run_typing_activity.py

import numpy as np
import cv2
from src.detection.typing_activity import TypingActivityDetector

def main():
    # Durée (s) pendant laquelle on continue d'afficher « typing... » après la dernière touche
    detector = TypingActivityDetector(display_timeout=0.5)
    detector.start()

    win = 'Typing Indicator'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 300, 100)

    try:
        while True:
            # Création d'une image blanche de 100×300 px
            img = 255 * np.ones((100, 300, 3), dtype=np.uint8)

            # Si on vient d'appuyer sur une touche...
            if detector.is_typing():
                cv2.putText(
                    img, 'typing...', (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )

            cv2.imshow(win, img)

            # Quitter sur 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

    finally:
        detector.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
