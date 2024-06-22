import cv2
from skeleton_points import Skeleton
def farme_extract(video_path , frame_interval = 8):
  FRAMES = []


  cap = cv2.VideoCapture(video_path)


  frame_count = 0
  while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame_count+=1
    if frame_count%frame_interval==0:

      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      FRAMES.append(gray_frame)
  #print(frame_count)
  cap.release()
  #print(len(FRAMES))
  return FRAMES

if __name__=='__main__':
   farme_extract('lift__V1-0003.mov')

