import mediapipe as mp
import numpy as np
import torch
class Skeleton:
  def __init__(self):
    self.mp_pose = mp.solutions.pose
    self.pose = self.mp_pose.Pose()
    self.mp_drawing = mp.solutions.drawing_utils

  def point_extractor(self,frames):
    skeleton_points=[]
    for frame in frames:
      results = self.pose.process(frame)

      if results.pose_landmarks:
          frame_pose_points=[]
          for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx>10:
                frame_pose_points.append(landmark.x)
                frame_pose_points.append(landmark.y)
                frame_pose_points.append(landmark.z)


                # print(f"X={landmark.x} Y={landmark.y} Z={landmark.z}")
               
          skeleton_points.append(frame_pose_points)
          
          # skeleton_points=skeleton_points.flatten()
    skeleton_points =torch.tensor(skeleton_points)
    return skeleton_points         
          
  def draw_frames(self,frames,model=None):
      final_frames=[]

      for frame in frames:
        results = self.pose.process(frame)
        image_rgb = frame
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            final_frames.append(image_rgb)
      
      return final_frames    

            

                  
            

  