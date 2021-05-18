import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio 
import numpy as np 
from sklearn.model_selection import train_test_split
import os 
from scipy.fftpack import fft
import random
import gc
import matplotlib.pyplot as plt

# load all data
yourPath = 'C:/Users/Marco/Desktop/研究/CWRU/12k/full_data/'
allFileList = os.listdir(yourPath)

data_total_len = 120000
data_len = 400
data_num = data_total_len//data_len
x_fft_len = 200
load_matrix = np.zeros([len(allFileList),data_total_len])
load_0hp_list = [0,1,2,16,17,18,32,44,45,46]
load_1hp_list = [4,5,6,20,21,22,33,47,48,49]
load_2hp_list = [8,9,10,24,25,26,34,50,51,52]
load_3hp_list = [12,13,14,28,29,30,35,53,54,55]  #53是08?
class_num = 10

def data_process(load_data):
    
    load_data_value = load_data.keys()

    for s in load_data_value:
        if 'DE' in s:
            load_matrix = load_data[s][500:data_total_len+500]
    
    return load_matrix
def load_hp_data(a_list):
    
    
    load_hp_data_mat = np.zeros([len(a_list),data_total_len])
    for i, file in enumerate(a_list):
        
        load_hp_data_mat[i,:] = load_matrix[file]
        
    return load_hp_data_mat
def x_y_split(load_xhp):
    
    load_x = load_xhp.reshape([10*data_total_len//data_len,data_len])
    # load_x = np.abs(fft(load_x))[:,0:x_fft_len]
    
    load_y = np.zeros(data_num*class_num)
    
    for load_y_i in np.arange(load_y.shape[0]):
              
        load_y[load_y_i] = load_y_i//data_num
    
    load_x = torch.from_numpy(load_x)
    load_y = torch.from_numpy(load_y)

    return load_x, load_y

def image_prepare(images, labels):
    images = images.to(device)
    labels = labels.to(device)
    # print(images.shape)
    
    # unsqeeze
    images = torch.unsqueeze(images, 1)
    # labels = torch.unsqueeze(labels, 1)
 
    images = images.float()
    labels = labels.long()
    return images, labels

for i, file in enumerate(allFileList):
    
    load_data = sio.loadmat(yourPath+file)
    
    load_matrix[i,:] = data_process(load_data).reshape([data_total_len])
    
load_0hp = load_hp_data(load_0hp_list)
load_1hp = load_hp_data(load_1hp_list)
load_2hp = load_hp_data(load_2hp_list)
load_3hp = load_hp_data(load_3hp_list)


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 50
num_classes = 10
batch_size = 500
learning_rate = 0.001
source_x, source_y = x_y_split(load_2hp)
target_x, target_y = x_y_split(load_0hp)

target_x, target_y = image_prepare(target_x, target_y)
target_x_train, target_x_test, target_y_train, target_y_test = train_test_split(target_x, target_y, test_size=1/6, random_state=5, shuffle=True)

source_x, source_y = image_prepare(source_x, source_y)
source_x_train, source_x_test, source_y_train, source_y_test = train_test_split(source_x, source_y, test_size=1/6, random_state=4, shuffle=True)


# Convolutional neural network (two convolutional layers)
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 50, kernel_size=50, stride=1, padding=25)
        self.conv2 = nn.Conv1d(50, 50, kernel_size=50, stride=1, padding=25)
        self.conv3 = nn.Conv1d(50, 50, kernel_size=50, stride=1, padding=25)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) 
        self.fc1 = nn.Linear(50*50, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(50)
        self.bn2 = nn.BatchNorm1d(50)
        self.bn3 = nn.BatchNorm1d(50)       
        self.drop = nn.Dropout(0.5)
          
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 50*50) #flattening
        x = self.fc1(x)
        self.featuremap = x
        x = self.fc2(x)
        return x

model = CNN().to(device)


# Loss and optimizer
def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def compute_mmd(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

CE_Loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
total_step = source_x_train.size(0)//batch_size

total = 0.0
correct = 0.0
running_loss = 0.0
loss_values = []
acc_value = []
for epoch in range(num_epochs):
       
    for i in range(total_step):
        mmd_num = 100
        random_num = random.randrange(source_x_train.size(0)-batch_size)
        
        model(target_x_train[random_num:random_num + mmd_num ,:,:])
        target_feature = model.featuremap.cpu()  
        
        model(source_x_train[random_num:random_num + mmd_num ,:,:])
        source_feature = model.featuremap.cpu()
        
        images = source_x_train[i*batch_size:+i*batch_size + batch_size,:,:]
        labels = source_y_train[i*batch_size:+i*batch_size + batch_size]

        outputs = model(images)
        loss = CE_Loss(outputs, labels) + compute_mmd(source_feature, target_feature)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print ('Epoch[{}/{}], Iteration[{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    pred = model(source_x_train)
    _ , prediction = torch.max(pred.data, 1)
    running_accuracy = (prediction == source_y_train).sum().item()/2500*100
    print ('Epoch[{}/{}] Accuracy: {:.2f} %' .format(epoch+1, num_epochs, running_accuracy))
    running_loss =+ loss.item()
    loss_values.append(running_loss)
    acc_value.append(running_accuracy)
    
plt.figure(figsize=(10, 3))     
plt.subplot(1, 2, 1) 
plt.plot(loss_values)
plt.title("Loss")
plt.subplot(1, 2, 2) 
plt.plot(acc_value)
plt.title("Accuracy")
plt.savefig("final.jpg")

# Save the model checkpoint
model.eval()
torch.save(model, 'CDA.pth')


# Test the model
model = torch.load('CDA.pth')
with torch.no_grad():
    correct = 0
    outputs = model(source_x_train)
    _ , predicted = torch.max(outputs.data, 1)
    correct = (predicted == source_y_train).sum().item()
    print('Source Domain Train Accuracy : {} %'.format(100 * correct / 2500))

with torch.no_grad():
    correct = 0
    outputs = model(source_x_test)
    _ , predicted = torch.max(outputs.data, 1)
    correct = (predicted == source_y_test).sum().item()
    print('Source Domain Test Accuracy : {} %'.format(100 * correct / 500))
    
with torch.no_grad():
    correct = 0
    outputs = model(target_x_train)
    _ , predicted = torch.max(outputs.data, 1)
    correct = (predicted == target_y_train).sum().item()
    print('Target Domain Train Accuracy : {} %'.format(100 * correct / 2500))
    
with torch.no_grad():
    correct = 0
    outputs = model(target_x_test)
    _ , predicted = torch.max(outputs.data, 1)
    correct = (predicted == target_y_test).sum().item()
    print('Target Domain Test Accuracy : {} %'.format(100 * correct / 500))    

