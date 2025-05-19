import cv2
import mediapipe as mp
import numpy as np

bg\_image = cv2.imread("office\_BG.jpg.jpg")

cap = cv2.VideoCapture(0)

mp\_selfie\_segmentation = mp.solutions.selfie\_segmentation
segment = mp\_selfie\_segmentation.SelfieSegmentation(model\_selection=1)

while True:
ret, frame = cap.read()
if not ret:
break

```
frame = cv2.flip(frame, 1)
bg_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = segment.process(rgb_frame)
mask = results.segmentation_mask > 0.5
output_image = np.where(mask[..., np.newaxis], frame, bg_resized)
cv2.imshow("Virtual Background (No Green Screen)", output_image)

if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

cap.release()
cv2.destroyAllWindows()
