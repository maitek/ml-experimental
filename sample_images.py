import cv2
import numpy as np

video_file = "clouds_real.mp4"

cap = cv2.VideoCapture(video_file)

output_size = (256,256)
samples_per_frame = 1

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")
i = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    #import pdb; pdb.set_trace()

    rs = np.random.randint(output_size[1],frame.shape[0])
    print(rs)
    rx = np.random.randint(0,frame.shape[0]-rs)
    ry = np.random.randint(0,frame.shape[1]-rs)
    frame_crop = frame[rx:rx+rs, ry:ry+rs,:]
    frame_crop = cv2.resize(frame_crop,output_size)

    if np.random.rand() > 0.5:
        frame_crop = np.flip(frame_crop,axis=1)

    if np.random.rand() > 0.5:
        frame_crop = np.flip(frame_crop,axis=0)

    # Display the resulting frame
    cv2.imshow('Frame',frame_crop)
    cv2.imwrite("real/frame_{}.jpg".format(i),frame_crop)
    cv2.waitKey(1)
    i+=1
