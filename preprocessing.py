import os 
from skeleton_points import Skeleton
from sklearn.model_selection import train_test_split
from frame_extracted import farme_extract
import torch
from tqdm import tqdm
from encoding import one_hot_encode
def preprocess(data_path,split=True):
    FEATURES = []
    LABELS = []

    pose_list = os.listdir(data_path)
    skeleton = Skeleton() 
    for pose in pose_list:
        pose_path = os.path.join(data_path , pose)
        videos  = os.listdir(pose_path)
        print(pose)
        for video in tqdm(videos , desc = 'processing' , unit ='video'):

            video_path = os.path.join(pose_path , video)
            FRAMES = farme_extract(video_path)
            
            skeleton_points = skeleton.point_extractor(FRAMES)
            

           # print(len(FRAMES))
            if len(skeleton_points)>=14:
                skeleton_points = skeleton_points[:14]
                FEATURES.append(skeleton_points)
                LABELS.append(pose)
    # sum = 0
    # mn=100        
    # for feature in FEATURES:
    #     sum+=len(feature)
    #     mn = min(mn,len(feature))
    # print(sum/len(FEATURES))
    # print(mn)
         
    FEATURES=torch.tensor(FEATURES)
    LABELS = one_hot_encode(LABELS)   

    # return x_train, x_test ,
    X = FEATURES
    Y = LABELS
    if split:
        X_train , X_test , Y_train , Y_test =train_test_split(X,Y,test_size=0.3,shuffle=True)
        X_train , X_test , Y_train , Y_test = torch.tensor(X_train,dtype=torch.float32) , torch.tensor(X_test,dtype=torch.float32 ), torch.tensor(Y_train,dtype= torch.float32 ) , torch.tensor(Y_test,dtype=torch.float32)
        print(X_train.shape , X_test.shape , Y_train.shape , Y_test.shape)
        return  X_train , X_test , Y_train , Y_test 
    
    return FEATURES , LABELS
     

           