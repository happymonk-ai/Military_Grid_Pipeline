import cv2
import numpy as np
import torch
import time

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='three_class_05_dec.pt')

# RTSP URLs
urls = ['rtsp://happymonk:admin123@streams.ckdr.co.in:1554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
        'rtsp://test:test123456789@streams.ckdr.co.in:2554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
        'rtsp://happymonk:admin123@streams.ckdr.co.in:3554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
        'rtsp://happymonk:admin123@streams.ckdr.co.in:4554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
        'rtsp://happymonk:admin123@streams.ckdr.co.in:5554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
        'rtsp://happymonk:admin123@streams.ckdr.co.in:6554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
        'rtsp://happymonk:admin123@streams.ckdr.co.in:6554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
        'rtsp://happymonk:admin123@streams.ckdr.co.in:6554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif']

# Initialize cameras
cameras = []
for url in urls:
    cam = cv2.VideoCapture(url)
    cameras.append(cam)

# Create window
cv2.namedWindow('MultiCam', cv2.WINDOW_NORMAL)

while True:
    # Capture frames from each camera
    frames = []
    for cam in cameras:
        ret, frame = cam.read()
        if ret:
            frame = cv2.resize(frame, (1080, 720))
            frames.append(frame)
            print(frame.shape)
    
    # Process frames with YOLOv5
    t0 = time.time()
    processed_frames = []
    for frame in frames:
        # Detect objects using YOLOv5
        results1 = model(frame)
    
        # Loop through YOLOv5 predictions
        for pred in results1.pred:
            # Get prediction data
            xyxy = pred[:, :4].cpu().numpy()
            conf = pred[:, 4].cpu().numpy()
            class_ids = pred[:, 5].cpu().numpy().astype(int)
            
            # Draw bounding boxes and labels
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cls_id = class_ids[i]
                cls_name = model.names[cls_id]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        processed_frames.append(frame)
    t1 = time.time()
    fps = round(1 / (t1 - t0), 2)
    
    # Display the processed frames in the window
    rows = []
    for i in range(0, 8, 2):
        row = np.concatenate((processed_frames[i], processed_frames[i+1]), axis=1)
        rows.append(row)
    multi = np.concatenate(rows, axis=0)
    cv2.putText(multi, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('MultiCam', multi)
    
    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and destroy the window
for cam in cameras:
    cam.release()
cv2.destroyAllWindows()