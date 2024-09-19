import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the MiDaS model for depth estimation
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Input transformation pipeline for MiDaS
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Load YOLOv8 model using Ultralytics package
yolo_model = YOLO('yolov8n.pt')  # Load the YOLOv8n (nano) model

# Hook into OpenCV for live camera feed
cap = cv2.VideoCapture(0)

# Focal length for distance estimation (example value, you may need to adjust this)
FOCAL_LENGTH = 150  # Adjust based on your camera
KNOWN_HEIGHT = 70  # Average height of a person in inches

while cap.isOpened():
    ret, frame = cap.read()

    # Step 1: MiDaS depth prediction
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_batch = transform(img_rgb).to('cpu')

    with torch.no_grad():
        prediction = midas(img_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Step 2: YOLOv8 person detection
    results = yolo_model(frame)

    # Step 3: Filter for person detection only
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = box.cls[0].item()  # Class index

            if int(cls) == 0:  # YOLOv8 'person' class is typically labeled as 0
                # Bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Extract depth from the center of the bounding box
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                depth_value = depth_map[y_center, x_center]

                # Calculate distance using focal length formula: Distance = (Focal_Length * Real_Height) / Height_in_Image
                height_in_image = y2 - y1
                distance = (FOCAL_LENGTH * KNOWN_HEIGHT) / height_in_image if height_in_image > 0 else 0

                # Display distance on the frame
                cv2.putText(frame, f'Distance: {distance:.2f} inches', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Step 4: Display the live video feed
    cv2.imshow('Live Feed', frame)
    plt.imshow(depth_map, cmap='plasma')
    plt.pause(0.001)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.show()
