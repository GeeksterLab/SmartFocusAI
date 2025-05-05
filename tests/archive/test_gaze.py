import cv2
from gaze.face_mesh import FaceMeshDetector
from gaze.gaze_estimator import GazeEstimator

detector = FaceMeshDetector()
estimator = GazeEstimator(horiz_left_thresh=0.3,
                          horiz_right_thresh=0.7,
                          vert_up_thresh=0.3,
                          vert_down_thresh=0.7)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    lm = detector.process(frame)
    gaze = estimator.estimate(frame, lm)
    cv2.putText(frame, f"Regard: {gaze}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Gaze Test", frame)
    if cv2.waitKey(1) in (27, ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
