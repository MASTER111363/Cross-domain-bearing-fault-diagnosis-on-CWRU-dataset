import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import os 
from scipy.io import loadmat
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import random

# load all data
yourPath = 'CWRU/12k/full_data/'
allFileList = os.listdir(yourPath)

data_total_len = 120000
data_len = 600
data_num = 200
fft_len = data_len//2
class_num = 4

# data_len = 120000
# resample_len = 300
# data_len = 1200
# data_num = 400
# used_data_len = resample_len * data_num + data_len
# fft_len = data_len//2
# class_num = 4

def Load_hp(category):
    output_data = np.zeros((4, data_total_len))
    file = [name for name in allFileList if 
            (category in name)&('07'in name)&('outerrace06'not in name)&
            ('outerrace12'not in name) or (category in name)&('normal'in name)]
    file = sorted(file)
    print(file)
    for num, name in enumerate(file):
        data = loadmat(yourPath + name)
        points = [x for x in data.keys() if 'DE' in x]
        output_data[num, :] = data[points[0]][500:data_total_len+500].flatten()
    return output_data

load_0hp = Load_hp('0hp')
load_1hp = Load_hp('1hp')
load_2hp = Load_hp('2hp')
load_3hp = Load_hp('3hp')

def x_y_prepare(load_hp):    
    load_x = load_hp.reshape([class_num*data_num,data_len])
    # load_x = np.abs(fft(load_x))[:,0:fft_len]

    load_y = np.zeros(class_num*data_num)    
    for i in np.arange(load_y.shape[0]):
        load_y[i] = i//data_num
    # print(load_y)
    # load_x = load_x[:,10:60]
    load_x = torch.from_numpy(load_x)
    load_y = torch.from_numpy(load_y)
    return load_x, load_y

def image_prepare(images, labels):
    images = images.to(device)
    labels = labels.to(device)
    # unsqeeze
    # images = torch.unsqueeze(images, 1)
    # labels = torch.unsqueeze(labels, 1)
    images = images.float()
    labels = labels.long()
    return images, labels


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 50
batch_size = 64
source_data_num = 600
target_data_num = 600
learning_rate = 0.0001
source_x, source_y = x_y_prepare(load_3hp)
target_x, target_y = x_y_prepare(load_0hp)

target_x, target_y = image_prepare(target_x, target_y)
target_x_train, target_x_test, target_y_train, target_y_test = train_test_split(
    target_x, target_y, train_size=target_data_num/class_num/data_num, test_size=150/class_num/data_num, shuffle=True)
source_x, source_y = image_prepare(source_x, source_y)
source_x_train, source_x_test, source_y_train, source_y_test = train_test_split(
    source_x, source_y, train_size=source_data_num/class_num/data_num, test_size=150/class_num/data_num, shuffle=True)

### 0:ball ; 1:innerrace ; 2:normal ; 3:outerrace ###

plt.figure(figsize=(7, 3))   
plt.subplot(1, 1, 1) 
plt.plot(np.squeeze(source_x_train[:1,:].cpu()))
plt.title("Raw data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# plt.figure(figsize=(20, 6.5))   
# plt.subplot(2, 3, 1) 
# plt.bar(range(100), np.squeeze(source_x_train[:1,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[:1].cpu()))
# plt.subplot(2, 3, 2) 
# plt.bar(range(100), np.squeeze(source_x_train[1:2,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[1:2].cpu()))
# plt.subplot(2, 3, 3) 
# plt.bar(range(100), np.squeeze(source_x_train[2:3,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[2:3].cpu()))
# plt.subplot(2, 3, 4) 
# plt.bar(range(100), np.squeeze(source_x_train[3:4,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[3:4].cpu()))
# plt.subplot(2, 3, 5) 
# plt.bar(range(100), np.squeeze(source_x_train[4:5,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[4:5].cpu()))
# plt.subplot(2, 3, 6) 
# plt.bar(range(100), np.squeeze(source_x_train[5:6,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[5:6].cpu()))
# plt.savefig("source2.jpg")


# plt.figure(figsize=(12, 6.5))   
# plt.subplot(2, 3, 1) 
# plt.bar(range(100), np.squeeze(target_x_train[:1,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[:1].cpu()))
# plt.subplot(2, 3, 2) 
# plt.bar(range(100), np.squeeze(target_x_train[1:2,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[1:2].cpu()))
# plt.subplot(2, 3, 3) 
# plt.bar(range(100), np.squeeze(target_x_train[2:3,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[2:3].cpu()))
# plt.subplot(2, 3, 4) 
# plt.bar(range(100), np.squeeze(target_x_train[3:4,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[3:4].cpu()))
# plt.subplot(2, 3, 5) 
# plt.bar(range(100), np.squeeze(target_x_train[4:5,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[4:5].cpu()))
# plt.subplot(2, 3, 6) 
# plt.bar(range(100), np.squeeze(target_x_train[5:6,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[5:6].cpu()))
# plt.savefig("target2.jpg")


# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         self.nn1 = nn.Linear(fft_len, 100)         
#         # self.nn2 = nn.Linear(200, 50)
#         self.nn3 = nn.Linear(100, class_num)    
#         self.leakyrelu = nn.LeakyReLU()

#     def forward(self, x):
#         x = self.nn1(x)
#         x = self.leakyrelu(x)
#         # x = self.nn2(x)       
#         # x = self.leakyrelu(x)
#         self.featuremap = x
#         x = self.nn3(x)
#         x = F.softmax(x,dim=1)
#         return x

# model = NN().to(device)


# # Loss and optimizer
# class MMD_loss(nn.Module):
#     def __init__(self, kernel_mul = 2.0, kernel_num = 5):
#         super(MMD_loss, self).__init__()
#         self.kernel_num = kernel_num
#         self.kernel_mul = kernel_mul
#         self.fix_sigma = None
        
#     def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#         n_samples = int(source.size()[0])+int(target.size()[0])
#         total = torch.cat([source, target], dim=0)

#         total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         L2_distance = ((total0-total1)**2).sum(2) 
#         if fix_sigma:
#             bandwidth = fix_sigma
#         else:
#             bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
#         bandwidth /= kernel_mul ** (kernel_num // 2)
#         bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
#         kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
#         return sum(kernel_val)

#     def forward(self, source, target):
#         # batch_size = int(source.size()[0] + target.size()[0])//2
#         batch_size = int(source.size()[0])
#         kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
#         XX = kernels[:batch_size, :batch_size]
#         YY = kernels[batch_size:, batch_size:]
#         XY = kernels[:batch_size, batch_size:]
#         YX = kernels[batch_size:, :batch_size]
#         # loss = torch.mean(XX + YY - XY -YX)
#         loss = torch.mean(XX)
#         loss += torch.mean(YY)
#         loss -= torch.mean(XY)
#         loss -= torch.mean(YX)
#         return loss

# CE_Loss = nn.CrossEntropyLoss()
# MMD = MMD_loss().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# # Train the model
# total_step = source_x_train.size(0)//batch_size
# start = time.time()
# total = 0.0
# correct = 0.0
# running_loss = 0.0
# loss_values = []
# acc_value = []

# for epoch in range(num_epochs):
#     model.train()
       
#     for i in range(total_step):
#         # mmd_num = 50
#         # random_num = random.randrange(target_x_train.size(0)-mmd_num)
        
#         images = source_x_train[i*batch_size:+i*batch_size + batch_size,:]
#         labels = source_y_train[i*batch_size:+i*batch_size + batch_size]
#         outputs = model(images)
#         source_feature = model.featuremap.to(device)       
        
#         # model(target_x_train[random_num:random_num + mmd_num ,:])
#         # target_feature = model.featuremap.to(device) 

#         loss = CE_Loss(outputs, labels) #+ MMD(source_feature, target_feature)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         print ('Epoch[{}/{}], Iteration[{}/{}], Loss: {:.4f}' 
#                     .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
#     pred = model(source_x_train)
#     _ , prediction = torch.max(pred.data, 1)
#     running_accuracy = (prediction == source_y_train).sum().item()/source_x_train.size(0)*100
#     print ('Epoch[{}/{}] Accuracy: {:.2f} %' .format(epoch+1, num_epochs, running_accuracy))
#     running_loss =+ loss.item()
#     loss_values.append(running_loss)
#     acc_value.append(running_accuracy)
    
# # plt.figure(figsize=(10, 3))     
# # plt.subplot(1, 2, 1) 
# # plt.plot(loss_values)
# # plt.title("Loss")
# # plt.subplot(1, 2, 2) 
# # plt.plot(acc_value)
# # plt.title("Accuracy")
# # plt.savefig("final.jpg")
# end = time.time()
# print('training time: {:.3f} s'.format(end-start))

# def get_num_parameters(model):
#     pp=0
#     for p in list(model.parameters()):
#         nn=1
#         for s in list(p.size()):
#             nn = nn*s
#         pp += nn
#     return pp
# print('number of parameters: ', get_num_parameters(model))


# # Save the model checkpoint
# torch.save(model, 'CDA.pth')
# torch.save(model.state_dict(), 'CDA_weight.pth')


# # Test the model
# model = NN().to(device)
# model.load_state_dict(torch.load('CDA_weight.pth'))
# # model = torch.load('CDA.pth', map_location=torch.device('cpu'))
# model.eval()

# def evaluate(data, label):
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         outputs = model(data)
#         _ , predicted = torch.max(outputs.data, 1)
#         correct = (predicted == label).sum()
#         print('{:.2f} %'.format(100 * correct / data.size(0)))     
#         print(data.size(0))

# print('Source Domain Training Accuracy : ', end = '')
# evaluate(source_x_train, source_y_train)
# print('Source Domain Testing Accuracy : ', end = '')
# evaluate(source_x_test, source_y_test)
# print('Target Domain Training Accuracy : ', end = '')
# evaluate(target_x_train, target_y_train)
# print('Target Domain Testing Accuracy : ', end = '')
# evaluate(target_x_test, target_y_test)


# def plot_with_labels(lowDWeights, labels, text):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         color = cm.rainbow(int(255 * s / 13))
#         plt.text(x, y, s, backgroundcolor=color, fontsize=5)
#     plt.xlim(X.min(), X.max()) 
#     plt.ylim(Y.min(), Y.max()) 
#     plt.title('Visualized fully connected layer')
#     plt.savefig("TSNE_{}.jpg".format(text))

# # tsne = TSNE(perplexity=30, n_components=2, init='pca')
# # plot_only = 200
# # model(source_x_train)
# # source = model.featuremap.cpu().detach().numpy()[:plot_only, :]
# # model(target_x_train)
# # target = model.featuremap.cpu().detach().numpy()[:plot_only, :]
# # CCC = np.vstack((source,target))
# # low_dim_embs = tsne.fit_transform(CCC)
# # labels_source = source_y_train.cpu().numpy()[:plot_only]
# # labels_target = target_y_train.cpu().numpy()[:plot_only]
# # YYY = np.hstack((labels_source,labels_target+10))
# # plot_with_labels(low_dim_embs, YYY, 'merge')




