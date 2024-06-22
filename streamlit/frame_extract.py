import cv2
import numpy as np
from skeleton import Skeleton
from tqdm import tqdm
def farme_extract(video_path , frame_interval = 8):
  FRAMES = []


  cap = cv2.VideoCapture(video_path)
  Tframe_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    

  frame_count = 0
  for i in tqdm(range(Tframe_count)):
    ret, frame = cap.read()

    if not ret:
        break
    frame_count+=1
    if frame_count%frame_interval==0:
      # print('frames extracted')
      # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      FRAMES.append(frame)
  #print(frame_count)
  cap.release()
  #print(len(FRAMES))
  FRAMES= np.array(FRAMES)
  return FRAMES

if __name__=='main_':
   farme_extract('lift__V1-0003.mov')