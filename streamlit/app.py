import streamlit as st
import cv2
from PIL import Image
import torch
from model.model import PRNN
from frame_extract import farme_extract
from skeleton import Skeleton
import numpy as np
import subprocess
def eval(X_test,model):
    with torch.no_grad():
        test_data=test_data= X_test.unsqueeze(0)
        output = model(test_data)
        output = [[1 if val > 0.5 else 0 for val in row] for row in output]
        output=np.array(output)
        output= np.squeeze(output)

    return output

def output_write(frames , outputs):
    idx = 0
    dicti = ["smash" ,"lift", "clear" ]
    output_frame=[]
    for id ,  frame in enumerate(frames):
        if id==0 or id%112!=0:
            for j, i in enumerate(outputs[idx]):
                if i==1:
                    temp=frame
                    cv2.putText(temp, dicti[j], (50, 300), cv2.FONT_HERSHEY_SIMPLEX,5, (148,0,211), 20, cv2.LINE_AA)
                    output_frame.append(temp)
       
        else:
            idx+=1
            if idx>= len(outputs):
                break
            
    output_frame+=frames[len(output_frame):]
    return output_frame

def video_write(frames,output_video_path='output.mp4'):
    print(len(frames))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' or 'MJPG'
    height,width,channel=frames[0].shape
    frame_size=(width,height)
    print(width, height)
    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, 62.5, (width, height))  # 24.0 is the frames per second (fps)

    # Write each frame to the video
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()



def main(model, skeleton):
    st.title('Video Frame Extractor')
    
    # Upload video
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    
    if uploaded_file is not None:
        # Display video
        video_bytes = uploaded_file.read()
        st.video(video_bytes)
        
        # Extract frames
      
        if st.button("start prediction"):
            # Save video to a temporary file
            with open("temp_video.mp4", "wb") as f:
                f.write(video_bytes)
                
            # Extract frames
            frames = farme_extract("temp_video.mp4")
            print('training frames extracted',frames.shape)
            total_frames = farme_extract("temp_video.mp4",frame_interval=1)
            print('all frames extracted')
            final_frames=skeleton.draw_frames(total_frames)
            print('all frames drawn')
            total_frames=np.array(total_frames)
            
            # print(f"1->>frame size{final_frames.shape}")
            training_frames= frames[:-(len(frames)%14)]
            print(training_frames.shape)
            training_frames=training_frames.reshape(-1,14,*training_frames.shape[1:])
            print('going for skeleton points',)
            output=[]
            for frame_collection in training_frames:
                skeleton_points=skeleton.point_extractor(frame_collection)
                print(skeleton_points.shape)
                out=eval(skeleton_points,model)
                output.append(out)
            # skeleton_points=skeleton.point_extractor(training_frames)
            # print(skeleton_points.shape)
            # out=eval(skeleton_points,model)
            # output.append(out)
            final_frames=output_write(final_frames,outputs=output)
            final_frames=np.array(final_frames)
            video_write(final_frames)
            st.subheader(f"pridicted output video ")
            subprocess.run('ffmpeg -i output.mp4 -vcodec libx264 -y final_video.mp4',shell=True)
            st.video('final_video.mp4')
            # for i, frame in enumerate(training_frames):
            #     st.image(Image.fromarray(frame), caption=f"Frame {i+1}", use_column_width=True)


if __name__ == "__main__":
    skeleton=Skeleton()
    model=PRNN() 
    model.load_state_dict(torch.load('pretrained_model/model.pth'))
    st.title("posture recognition site")
    main(model=model,skeleton=skeleton)
