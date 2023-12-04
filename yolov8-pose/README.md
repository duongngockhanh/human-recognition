# YOLOv8 Pose

16 keypoints

```
!pip install ultralytics
```
```
from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
```
Inference
```
results = model("a_standing_image.jpg")
for result in results:
  keypoints = result.keypoints
```
Predict into video
```
model.predict('a_standing_image.jpg', save=True, imgsz=320, conf=0.5)
```

## Input Image
![a_standing_image](https://github.com/duongngockhanh/human-recognition/assets/87640587/b07e515b-ae52-4a67-a042-592409f9750a)


## Output Image
![a_standing_image](https://github.com/duongngockhanh/human-recognition/assets/87640587/7efc2c9c-968a-4b54-9006-27b4ca79d97e)
