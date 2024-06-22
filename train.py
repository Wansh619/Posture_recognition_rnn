import os 
import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim
from models import PRNN
from preprocessing import preprocess
from tqdm import tqdm


def predict(X_test,model):
    c = 0
    with torch.no_grad():
        POSE_LIST=[]
        for idx, test in enumerate(X_test):
            test_data= test.unsqueeze(0)
            output = model(test_data)
            output = [[1 if val > 0.5 else 0 for val in row] for row in output]
            
            output=torch.tensor(output)
            POSE_LIST.append(output[0])
    return POSE_LIST 



def eval(X_test,Y_test,model):
    c = 0
    with torch.no_grad():
        for idx, test in enumerate(X_test):
            test_data= test.unsqueeze(0)
            output = model(test_data)
            output = [[1 if val > 0.5 else 0 for val in row] for row in output]
            output=torch.tensor(output)
            x=0
            for id, i in enumerate(Y_test[idx]):
                if i== output[0][id]:
                    x+=1

            if x==3:  
                c+=1
    return (c/len(Y_test))*100    

def train(folder_path,num_epoc=50):
    model=PRNN(input_size=66,hidden_size=256,num_layers=2,output_size=3)
    
    X_train , X_test , Y_train , Y_test  = preprocess(folder_path)
    learning_rate=0.001
    criteria = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(),lr=learning_rate)
    for i in tqdm(range(num_epoc)):
        for idx,feature in enumerate(X_train):
            reshape_data= feature.unsqueeze(0)
            pose=Y_train[idx].unsqueeze(0)
            # print(reshape_data.shape,pose.shape)
            score= model(reshape_data)
            loss=criteria(score,pose)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    print(eval(X_test,Y_test,model))
    save_path = 'model.pth'
    torch.save(model.state_dict(), save_path)        


    



train('Clips')

