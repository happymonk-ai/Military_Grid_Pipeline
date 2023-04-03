import cv2
import numpy as np
import torch
import time
results1_lst = []
# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='three_class_05_dec.pt')

# RTSP URLs
urls = ["rtsp://happymonk:admin123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://happymonk:admin123@192.168.1.2:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://test:test123456789@192.168.1.2:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://happymonk:admin123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif4",
        "rtsp://happymonk:admin123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://happymonk:admin123@192.168.1.2:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://test:test123456789@192.168.1.2:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://happymonk:admin123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif4",
        "rtsp://happymonk:admin123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://happymonk:admin123@192.168.1.2:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://test:test123456789@192.168.1.2:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://happymonk:admin123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif4",
        "rtsp://happymonk:admin123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://happymonk:admin123@192.168.1.2:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://test:test123456789@192.168.1.2:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        "rtsp://happymonk:admin123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif4",]

# Initialize cameras
cameras = []
for url in urls:
    cam = cv2.VideoCapture(url)
    cameras.append(cam)

# # Initialize cameras
# cameras = []
# for url in urls:
#     pipeline = "rtspsrc location={url} ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink".format(url = url)
#     cam = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
#     cameras.append(cam)

# Create window
cv2.namedWindow('MultiCam', cv2.WINDOW_NORMAL)
j = 0
while True:
    # Capture frames from each camera
    
    j +=1
    
    frames = []
    for cam in cameras:
        ret, frame = cam.read()
        
        frame = cv2.resize(frame, (1080, 720))
        frames.append(frame)
        print(frame.shape)
    # if j>20:
    # Process frames with YOLOv5
    t0 = time.time()
    processed_frames = []
    if j % 10 == 0 or j == 1:
        results1_lst = []
        for frame in (frames):
            # results1_lst = []
            print("entered")
            # Detect objects using YOLOv5
            
            results1 = model(frame)
            results1_lst.append(results1)
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
        print(len(results1_lst))
    else:
        for k,frame in enumerate(frames):
            # Loop through YOLOv5 predictions
            # for results1 in results1_lst: 
            for pred in results1_lst[k].pred:
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
    # print(len(processed_frames))
    t1 = time.time()
    # fps = round(1 / (t1 - t0), 2)
    
    # Display the processed frames in the window
    rows = []
    for i in range(0, 16, 4):
        # if len(processed_frames) == 4:
        row = np.concatenate((processed_frames[i], processed_frames[i+1],processed_frames[i+2],processed_frames[i+3]), axis=1)
        rows.append(row)

    multi = np.concatenate(rows, axis=0)
    # cv2.putText(multi, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('MultiCam', multi)
    
    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and destroy the window
for cam in cameras:
    cam.release()
cv2.destroyAllWindows()
