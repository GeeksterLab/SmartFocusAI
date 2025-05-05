import cv2
from gaze.face_mesh import FaceMeshDetector

def test_face_mesh():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    detector = FaceMeshDetector()
    window_name = "Test FaceMesh"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Détection des landmarks
        landmarks = detector.process(frame)

        # 2. Si visage détecté, récupérer et dessiner les iris
        if landmarks:
            iris_pts = detector.get_iris_points(frame, landmarks)
            for (x, y) in iris_pts["left_iris"] + iris_pts["right_iris"]:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # 3. Affichage
        cv2.imshow(window_name, frame)

        # 4. Lecture de la touche (1 ms)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key in (ord('q'), ord('Q')):
            # Échap (27) ou q/Q
            break

        # 5. Sortie si l'utilisateur a fermé la fenêtre (croix)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_face_mesh()
