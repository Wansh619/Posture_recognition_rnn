import  torch
import torch.nn as nn
import torch.optim as optim

class PRNN(nn.Module):
    def __init__(self, input_size=66, hidden_size=256 ,num_layers=2,output_size=3):
        super(PRNN,self).__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.output_size=output_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.relu=nn.ReLU()
        self.linear=nn.Linear(hidden_size,output_size)


    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        out,_=self.lstm(x,(h0,c0))
        out=self.relu(out)
        out=self.linear(out[:,-1,:])
        return out



# if _name=='main_':






#     label=torch.tensor([[0.003]*3]*1)
#     data=torch.zeros((1,14,66))
#     print(data.shape,data.dim())
#     model=PRNN(input_size=66,hidden_size=256,num_layers=3,output_size=3)
#     print(model)
#     learning_rate=0.001
#     criteria=nn.CrossEntropyLoss()
#     optimiser=optim.Adam(model.parameters(),lr=learning_rate)
#     for i in range(2):
#         # for (features,label) in data:
#         # data= features.reshape((-1,features.shape))
#         score= model(data)
#         print(score)
#         loss=criteria(score,label)
#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()