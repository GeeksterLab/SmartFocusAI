import cv2

print("Test de capture caméra\n")

for idx in range(4):
    # essaie sans forcer de backend
    cap = cv2.VideoCapture(idx)
    ret, frame = cap.read()
    print(f"Index {idx}: ret={ret}, frame ok? {frame is not None} ", 
          f"shape={None if frame is None else frame.shape}")
    cap.release()

cv2.destroyAllWindows()
