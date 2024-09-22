# author: ALEXIS CARBILLET

## import librairies
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.nn as nn
import torch
from sklearn.metrics import log_loss, roc_auc_score
from dateutil.parser import parse
import datetime

## import data
data = open("datatraining.txt", "r")
df=pd.DataFrame(index=range(8143), columns=["date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"])
data.readline()
for i in range(8143):
    l=data.readline().rstrip()
    l=l.split(',')
    l=l[1:]
    for j in range(len(l)):
        df.iloc[i][j]=l[j]
# print(df)

## preprocessing data
def dateProcessing(train):
    for i in range(train.shape[0]):
        dt = train[i].rstrip('"')
        dt = dt.lstrip('"')
        dt = parse(dt)
        train[i]=(dt-datetime.datetime(1970,1,1)).total_seconds() # convert to int
    text_merge=""
    
dateProcessing(df["date"])
labels=df["Occupancy"] # let's predict the occupancy
df=df.drop(["Occupancy"],axis=1)

(n,p)=df.shape

for i in range(n):
    for j in range(p):
        df.iloc[i][j]=float(df.iloc[i][j])
        labels.iloc[i]=float(labels.iloc[i])
x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.70, random_state=42)
        
x_train=torch.Tensor(list(x_train.values))
x_test=torch.Tensor(list(x_test.values))
y_train=torch.Tensor(list(y_train.values))
y_test=torch.Tensor(list(y_test.values))


## Multi-Layer Perceptron

class MLPModel(nn.Module):
    def __init__(self, num_features, dropout=0.25):
        super().__init__()
        self.drop=nn.Dropout(dropout)
        self.fc1=nn.Linear(num_features,4)
        self.fc2=nn.Linear(4,2)
        self.fc3=nn.Linear(2,1)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.drop(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.drop(x)
        x=self.fc3(x)
        # x=self.relu(x)
        return x
      
print('MLP')
model = MLPModel(6, dropout=0.25)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
model.eval()
model = MLPModel(6, dropout=0.25)
y_pred = model(x_test)
y=torch.Tensor([0]*len(y_pred))
mini=min(y_pred)
maxi=max(y_pred)
for i in range(len(y_pred)):
    if(maxi>mini):
        y[i]=(y_pred[i]-mini)/(maxi-mini) # normalization of the data, needed for the calculus of the loss
    else:
        print('maximum = minimum in y_pred')
        y[i]=0
before_train = criterion(y.squeeze(), y_test)
print('Test loss before training' , before_train.item())



model.train()
epoch = 7
for epoch in range(epoch):    
    optimizer.zero_grad()    # Forward pass
    y_pred = model(x_train)    # Compute Loss
    y=torch.Tensor([0]*len(y_pred))
    mini=min(y_pred)
    maxi=max(y_pred)
    for i in range(len(y_pred)):
        if(maxi>mini):
            y[i]=(y_pred[i]-mini)/(maxi-mini) # normalization of the data, needed for the calculus of the loss
        else:
            print('maximum = minimum in y_pred')
            y[i]=0
    
    loss = criterion(y.squeeze(), y_train)
   
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    # Backward pass
    loss.backward()
    optimizer.step()
    
model.eval()
y_pred = model(x_test)
y=torch.Tensor([0]*len(y_pred))
mini=min(y_pred)
maxi=max(y_pred)
for i in range(len(y_pred)):
    if(maxi>mini):
        y[i]=(y_pred[i]-mini)/(maxi-mini) # normalization of the data, needed for the calculus of the loss
    else:
        print('maximum = minimum in y_pred')
        y[i]=0
after_train = criterion(y.squeeze(), y_test) 
print('Test loss after Training' , after_train.item())

## Fully Convolutional Networks
        
class FCNModel(nn.Module):
    def __init__(self, num_features, dropout=0.25):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Conv1d(in_channels=6,out_channels=4,kernel_size=1)
        self.fc2 = nn.Conv1d(in_channels=4,out_channels=2,kernel_size=1)
        self.fc3 = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(2)
        

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x
        
print('FCN')
x_test.unsqueeze_(-1)
x_train.unsqueeze_(-1)
model = FCNModel(6, dropout=0.25)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
model.eval()
y_pred = model.forward(x_test)
y=torch.Tensor([0]*len(y_pred))
mini=min(y_pred)
maxi=max(y_pred)
for i in range(len(y_pred)):
    if(maxi>mini):
        y[i]=(y_pred[i]-mini)/(maxi-mini) # normalization of the data, needed for the calculus of the loss
    else:
        print('maximum = minimum in y_pred')
        y[i]=0
before_train = criterion(y.squeeze(), y_test)
print('Test loss before training' , before_train.item())

model.train()
epoch = 7
for epoch in range(epoch):    
    optimizer.zero_grad()    # Forward pass
    y_pred = model(x_train)    # Compute Loss
    y=torch.Tensor([0]*len(y_pred))
    mini=min(y_pred)
    maxi=max(y_pred)
    for i in range(len(y_pred)):
        if(maxi>mini):
            y[i]=(y_pred[i]-mini)/(maxi-mini) # normalization of the data, needed for the calculus of the loss
        else:
            print('maximum = minimum in y_pred')
            y[i]=0
    
    loss = criterion(y.squeeze(), y_train)
   
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    # Backward pass
    loss.backward()
    optimizer.step()

model.eval()
y_pred = model(x_test)
y=torch.Tensor([0]*len(y_pred))
mini=min(y_pred)
maxi=max(y_pred)
for i in range(len(y_pred)):
    if(maxi>mini):
        y[i]=(y_pred[i]-mini)/(maxi-mini) # normalization of the data, needed for the calculus of the loss
    else:
        print('maximum = minimum in y_pred')
        y[i]=0
after_train = criterion(y.squeeze(), y_test) 
print('Test loss after Training' , after_train.item())
