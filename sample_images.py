import cv2
import numpy as np

video_file = "animatrix.mp4"
import os
cap = cv2.VideoCapture(video_file)

output_size = (256,256)
samples_per_video = 2000

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = np.linspace(0,length,samples_per_video)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")
i = 1998
# Read until video is completed
for idx in idxs:
#while(cap.isOpened()):
  # Capture frame-by-frame
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
  ret, frame = cap.read()
  #import pdb; pdb.set_trace()
  if ret == True:
    #import pdb; pdb.set_trace()
    frame = cv2.resize(frame,(640,360))
    frame = frame[:280,50:-50]

    #rs = np.random.randint(output_size[1],frame.shape[0])
    #ry = 400
    rs = frame.shape[0]

    rx = np.random.randint(0,frame.shape[0]-rs+1)
    ry = np.random.randint(0,frame.shape[1]-rs+1)
    print(rs,rx,ry)
    frame_crop = frame[rx:rx+rs, ry:ry+rs,:]
    frame_crop = cv2.resize(frame_crop,output_size)

    if np.random.rand() > 0.5:
        frame_crop = np.flip(frame_crop,axis=1)

    #if np.random.rand() > 0.5:
    #    frame_crop = np.flip(frame_crop,axis=0)

    # Display the resulting frame
    #cv2.imshow('Frame',frame_crop)
    out_folder = "data/matrix/train/B/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    cv2.imwrite(out_folder + "frame_{}.jpg".format(i),frame_crop)
    #cv2.waitKey(1)
    i+=1
