import cv2
import math
import time
import numpy as np
import torch
import pulsectl
import subprocess


def set_sink(sink_name):
    with pulsectl.Pulse('my-client') as pulse:
        for sink in pulse.sink_list():
            if sink.name == sink_name:
                pulse.default_set(sink)

def sound_alarm():
    set_sink('alsa_output.pci-0000_00_1f.3.hdmi-stereo')
    filename = "./alarm.wav"
    subprocess.run(["aplay", "-D", "plughw:0,3", filename])


def main():

    cameras = []
    frames = []

    with open("/home/orin123/Desktop/url_file.txt", "r") as my_file:
        urls = my_file.read().split("\n")

    model = torch.hub.load('./yolov5-master', 'custom', path='three_class_05_dec.pt', source="local")

    num_rows = int(math.sqrt(len(urls)))
    num_cols = math.ceil(len(urls) / num_rows)
    total = num_cols*num_rows

    blank=cv2.imread("/home/orin123/anomaly_alarm_detection1/blank.jpg")

    cameras = []
    for url in urls:
        if url not in ["", " "]:
            cam = cv2.VideoCapture(url)
            cameras.append(cam)
    j=0
    ele=[0]*len(urls)  
    
    cv2.namedWindow('MultiCam', cv2.WINDOW_NORMAL)

    while True:
        j += 1
        frames = []
        for i, cam in enumerate(cameras):
            ret, frame = cam.read()
            if not ret or cam.isOpened() is False:
                print("no frame")
                print(urls[i])
                frame = cv2.imread("/home/orin123/anomaly_alarm_detection1/no_signal.jpg")
                cam=cv2.VideoCapture(urls[i])
                cameras[i] = cam
            frame = cv2.resize(frame, (1080, 720))
            frames.append(frame)
            print(frame.shape)
            
        t0 = time.time()
        processed_frames = []

        if j % 25 == 0 or j == 1:
            results1_lst = []
            for count, frame in enumerate(frames):
                results1 = model(frame)
                results1_lst.append(results1)

                for pred in results1.pred:
                    xyxy = pred[:, :4].cpu().numpy()
                    conf = pred[:, 4].cpu().numpy()
                    class_ids = pred[:, 5].cpu().numpy().astype(int)

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        cls_id = class_ids[i]
                        cls_name = model.names[cls_id]
                        if cls_name == 'Elephant':
                            ele[count] += 1
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                processed_frames.append(frame)
                

        else:
            for k, frame in enumerate(frames):
                for pred in results1_lst[k].pred:
                    xyxy = pred[:, :4].cpu().numpy()
                    conf = pred[:, 4].cpu().numpy()
                    class_ids = pred[:, 5].cpu().numpy().astype(int)

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        cls_id = class_ids[i]
                        cls_name = model.names[cls_id]
                        if cls_name == 'Elephant':
                            ele[k] += 1
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(frame.shape)
                processed_frames.append(frame)


        if any(count>20 for count in ele):
            print("\n\n**********************Alarm***********************\n\n")
            sound_alarm()
            ele = [0]*len(urls)
            
        grid_img = None
        if len(processed_frames)<total:
            req = total - len(processed_frames)
            for i in range(req):
                processed_frames.append(blank)

        for i in range(num_rows): 
            row = None
            for j in range(num_cols): 
                index = i * num_cols + j
                if index < len(processed_frames):
                    if row is None:
                        row = processed_frames[index]
                    else:
                        row = cv2.hconcat([row, processed_frames[index]])
            if grid_img is None:
                grid_img = row
            else:
                grid_img = cv2.vconcat([grid_img, row])
    
        cv2.namedWindow("BAGDOGRA-ELEPHANT ALARM TRIGGER", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("BAGDOGRA-ELEPHANT ALARM TRIGGER", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('BAGDOGRA-ELEPHANT ALARM TRIGGER', grid_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    for cam in cameras:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()