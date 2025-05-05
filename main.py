# # main.py

# import cv2
# import AVFoundation
# import threading
# import time
# import sys
# from src.detection.typing_activity import TypingActivityDetector
# from src.gaze.face_mesh import FaceMeshDetector
# from src.gaze.blink_detector import BlinkDetector
# from src.gaze.gaze_estimator import GazeEstimator
# from src.gaze.smoother import DirectionSmoother
# from src.logic.focus_manager import FocusManager

# # Utiliser un Event pour synchroniser la demande de permission
# camera_permission_event = threading.Event()
# camera_permission_granted = False

# def request_camera_permission():
#     """Demande la permission d'accéder à la caméra et attend la réponse"""
#     global camera_permission_granted
    
#     def permission_callback(granted):
#         global camera_permission_granted
#         camera_permission_granted = granted
#         print(f"Camera permission granted: {granted}")
#         camera_permission_event.set()
    
#     print("Requesting camera permission...")
#     AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
#         AVFoundation.AVMediaTypeVideo,
#         permission_callback
#     )
    
#     # Attendre la réponse avec un timeout
#     permission_timeout = 10  # secondes
#     if not camera_permission_event.wait(permission_timeout):
#         print("Timeout waiting for camera permission response")
#         return False
    
#     return camera_permission_granted

# class CombinedGazeDetector:
#     """
#     Utilise deux GazeEstimator (horizontal + vertical) + smoothing
#     pour déterminer si l’utilisateur regarde le centre de l’écran.
#     """
#     def __init__(self, face_mesh: FaceMeshDetector):
#         self.fm = face_mesh

#         # 1) calibration horizontale (axis=0)
#         cal_h = {
#             "Right":  (-0.31, +0.03),
#             "Center": (+0.03, +0.17),
#             "Left":   (+0.17, +0.25),
#         }
#         self.ge_h = GazeEstimator(calibration_ranges=cal_h,
#                                   momentum=0.8, axis=0)

#         # 2) calibration verticale (axis=1)
#         cal_v = {
#             "Up":     (-float("inf"), -0.05),
#             "Center": (-0.05, -0.04),
#             "Down":   (-0.04, float("inf")),
#         }
#         self.ge_v = GazeEstimator(calibration_ranges=cal_v,
#                                   momentum=0.8, axis=1)

#         # smoothing
#         self.h_smoother = DirectionSmoother(window_size=5)
#         self.v_smoother = DirectionSmoother(window_size=5)

#     def is_gazing(self, frame) -> bool:
#         lm = self.fm.process(frame)
#         if lm is None:
#             return False

#         dir_h = self.ge_h.estimate(frame, lm) or "None"
#         dir_v = self.ge_v.estimate(frame, lm) or "None"
#         smoother_h = self.h_smoother.update(dir_h)
#         smoother_v = self.v_smoother.update(dir_v)

#         # Focus si l’un des axes est centré
#         return (smoother_h == "Center") or (smoother_v == "Center")


# def main():
#     # 1) Demande de permission caméra (synchrone)
#     if not request_camera_permission():
#         print("Erreur : permission d'accès à la caméra refusée.")
#         print("Vérifiez les paramètres de confidentialité dans Préférences Système > Confidentialité > Caméra")
#         return
    
#     # 2) Initialisations - seulement après avoir obtenu la permission
#     print("Initializing camera...")
#     # Essayer quelques tentatives pour ouvrir la caméra
#     max_attempts = 3
#     cap = None
    
#     for attempt in range(max_attempts):
#         cap = cv2.VideoCapture(1, apiPreference=cv2.CAP_AVFOUNDATION)
#         if cap.isOpened():
#             print(f"Camera opened successfully on attempt {attempt+1}")
#             break
#         else:
#             print(f"Attempt {attempt+1}/{max_attempts} failed, retrying in 1 second...")
#             time.sleep(1)
            
#     if not cap or not cap.isOpened():
#         print("Erreur : impossible d'ouvrir la caméra après plusieurs tentatives.")
#         print("Vérifiez que la caméra n'est pas utilisée par une autre application.")
#         return
    
#     # Récupérer et vérifier les propriétés de la caméra
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"Camera properties: {frame_width}x{frame_height} at {fps} FPS")
    
#     if frame_width <= 0 or frame_height <= 0:
#         print("Erreur : dimensions de caméra invalides. Vérifiez les pilotes de votre caméra.")
#         return
        
#     # Période d'échauffement pour stabiliser la caméra
#     print("Camera warm-up period (3 seconds)...")
#     warmup_frames = 0
#     warmup_start = time.time()
#     while time.time() - warmup_start < 3.0:  # 3 secondes d'échauffement
#         ret, frame = cap.read()
#         if ret and frame is not None and frame.size > 0:
#             warmup_frames += 1
#         time.sleep(0.05)  # Courte pause entre les lectures
    
#     print(f"Warm-up complete, captured {warmup_frames} frames")

#     # Typing
#     typing = TypingActivityDetector(display_timeout=0.5)
#     typing.start()

#     # FaceMesh pour Blink + Gaze
#     face_mesh = FaceMeshDetector(
#         max_num_faces=1,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )
#     blink = BlinkDetector()

#     # Combined gaze detector
#     gaze = CombinedGazeDetector(face_mesh)

#     # FocusManager
#     focus_mgr = FocusManager(typing_detector=typing,
#                              gaze_detector=gaze)

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret or frame is None or frame.size == 0:
#                 print("Erreur de capture: frame invalide ou vide") 
#                 # Tenter de récupérer après une erreur de lecture
#                 time.sleep(0.1)
#                 continue
                
#             # Vérifier que la frame a des dimensions raisonnables
#             if frame.shape[0] <= 10 or frame.shape[1] <= 10:
#                 print(f"Frame avec dimensions suspicieuses: {frame.shape}")
#                 continue

#             # 2) Blink update (incrément compteur si besoin)
#             lm = face_mesh.process(frame)
#             if lm is not None:
#                 blink.update(frame, lm, face_mesh)

#             # 3) Typing & Gaze
#             is_typing = typing.is_typing()
#             is_gazing = gaze.is_gazing(frame)
#             bl = blink.blink_count

#             # 4) Focus logic
#             focused = focus_mgr.is_focused(frame)

#             # 5) Affichage du statut
#             status = "Focused" if focused else "Distracted"
#             color  = (0,255,0) if focused else (0,0,255)
#             cv2.putText(frame, status, (10,30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
#             # 6) Affichage debug secondaire
#             cv2.putText(frame,
#                         f"Typing:{int(is_typing)} Center Gaze:{int(is_gazing)} Blink:{bl}",
#                         (10,60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                         (255,255,255), 1)

#             cv2.imshow("Smart Focus AI", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         typing.stop()
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



# main.py

import cv2
import AVFoundation
import threading
import time
import sys
from src.detection.typing_activity import TypingActivityDetector
from src.gaze.face_mesh import FaceMeshDetector
from src.gaze.blink_detector import BlinkDetector
from src.gaze.gaze_estimator import GazeEstimator
from src.gaze.smoother import DirectionSmoother
from src.logic.focus_manager import FocusManager

# Use an Event to synchronize the permission request
camera_permission_event = threading.Event()
camera_permission_granted = False

def request_camera_permission():
    """Request permission to access the camera and wait for the response."""
    global camera_permission_granted
    
    def permission_callback(granted):
        global camera_permission_granted
        camera_permission_granted = granted
        print(f"Camera permission granted: {granted}")
        camera_permission_event.set()
    
    print("Requesting camera permission...")
    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVFoundation.AVMediaTypeVideo,
        permission_callback
    )
    
    # Wait for the response with a timeout
    permission_timeout = 10  # seconds
    if not camera_permission_event.wait(permission_timeout):
        print("Timeout waiting for camera permission response")
        return False
    
    return camera_permission_granted

class CombinedGazeDetector:
    """
    Uses two GazeEstimators (horizontal + vertical) + smoothing
    to determine if the user is looking at the center of the screen.
    """
    def __init__(self, face_mesh: FaceMeshDetector):
        self.fm = face_mesh

        # 1) horizontal calibration (axis=0)
        cal_h = {
            "Right":  (-0.31, +0.03),
            "Center": (+0.03, +0.17),
            "Left":   (+0.17, +0.25),
        }
        self.ge_h = GazeEstimator(calibration_ranges=cal_h,
                                  momentum=0.8, axis=0)

        # 2) vertical calibration (axis=1)
        cal_v = {
            "Up":     (-float("inf"), -0.05),
            "Center": (-0.05, -0.04),
            "Down":   (-0.04, float("inf")),
        }
        self.ge_v = GazeEstimator(calibration_ranges=cal_v,
                                  momentum=0.8, axis=1)

        # smoothing
        self.h_smoother = DirectionSmoother(window_size=5)
        self.v_smoother = DirectionSmoother(window_size=5)

    def is_gazing(self, frame) -> bool:
        lm = self.fm.process(frame)
        if lm is None:
            return False

        dir_h = self.ge_h.estimate(frame, lm) or "None"
        dir_v = self.ge_v.estimate(frame, lm) or "None"
        smoother_h = self.h_smoother.update(dir_h)
        smoother_v = self.v_smoother.update(dir_v)

        # Focus if either axis is centered
        return (smoother_h == "Center") or (smoother_v == "Center")


def main():
    # 1) Camera permission request (synchronous)
    if not request_camera_permission():
        print("Error: camera access permission denied.")
        print("Check your privacy settings in System Preferences > Privacy > Camera")
        return
    
    # 2) Initializations – only after permission is granted
    print("Initializing camera...")
    # Try a few attempts to open the camera
    max_attempts = 3
    cap = None
    
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(1, apiPreference=cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"Camera opened successfully on attempt {attempt+1}")
            break
        else:
            print(f"Attempt {attempt+1}/{max_attempts} failed, retrying in 1 second...")
            time.sleep(1)
            
    if not cap or not cap.isOpened():
        print("Error: could not open the camera after several attempts.")
        print("Check that the camera is not being used by another application.")
        return
    
    # Retrieve and verify camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera properties: {frame_width}x{frame_height} at {fps} FPS")
    
    if frame_width <= 0 or frame_height <= 0:
        print("Error: invalid camera dimensions. Check your camera drivers.")
        return
        
    # Warm-up period to stabilize the camera
    print("Camera warm-up period (3 seconds)...")
    warmup_frames = 0
    warmup_start = time.time()
    while time.time() - warmup_start < 3.0:  # 3 seconds warm-up
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            warmup_frames += 1
        time.sleep(0.05)  # short pause between reads
    
    print(f"Warm-up complete, captured {warmup_frames} frames")

    # Typing detector
    typing = TypingActivityDetector(display_timeout=0.5)
    typing.start()

    # FaceMesh for Blink + Gaze
    face_mesh = FaceMeshDetector(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    blink = BlinkDetector()

    # Combined gaze detector
    gaze = CombinedGazeDetector(face_mesh)

    # FocusManager
    focus_mgr = FocusManager(typing_detector=typing,
                             gaze_detector=gaze)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Capture error: invalid or empty frame")
                # Attempt to recover after a read error
                time.sleep(0.1)
                continue
                
            # Check frame has reasonable dimensions
            if frame.shape[0] <= 10 or frame.shape[1] <= 10:
                print(f"Frame with suspicious dimensions: {frame.shape}")
                continue

            # 2) Blink update (increment counter if needed)
            lm = face_mesh.process(frame)
            if lm is not None:
                blink.update(frame, lm, face_mesh)

            # 3) Typing & Gaze
            is_typing = typing.is_typing()
            is_gazing = gaze.is_gazing(frame)
            bl = blink.blink_count

            # 4) Focus logic
            focused = focus_mgr.is_focused(frame)

            # 5) Display status
            status = "Focused" if focused else "Distracted"
            color  = (0,255,0) if focused else (0,0,255)
            cv2.putText(frame, status, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
            # 6) Secondary debug display
            cv2.putText(frame,
                        f"Typing:{int(is_typing)} Center Gaze:{int(is_gazing)} Blink:{bl}",
                        (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,255,255), 1)

            cv2.imshow("Smart Focus AI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        typing.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
