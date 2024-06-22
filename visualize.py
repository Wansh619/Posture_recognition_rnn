from frame_extracted import farme_extract
from skeleton_points import Skeleton
from models import PRNN
from preprocessing import preprocess
import torch
import cv2
def video_write(frames,output_video_path='output.mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' or 'MJPG'
    width, height=frames[0][0],frames[0][1]
    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (width, height))  # 24.0 is the frames per second (fps)

    # Write each frame to the video
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()

def vizualize(video_path):
    video_frames = farme_extract(video_path,frame_interval=1)
    data_frames,_=preprocess(video_path)
    skeleton  = Skeleton()
    skeleton_points=skeleton.point_extractor(data_frames)
    model=PRNN()
    model.load_state_dict(torch.load('model.pth'))
    output=predict(data_frames,model)

    final_frames = skeleton.draw_frames(frames , model)

