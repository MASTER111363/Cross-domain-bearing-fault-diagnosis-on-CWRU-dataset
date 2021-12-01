import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
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

# data_len = 120000
# resample_len = 300
# data_len = 1200
# data_num = 400
# used_data_len = resample_len * data_num + data_len
# fft_len = data_len//2
# class_num = 10

# def Load_hp(category):
#     output_data = np.zeros((10, used_data_len))
#     file = [name for name in allFileList if 
#             (category in name)&('outerrace12'not in name)&
#             ('outerrace03'not in name)&('28'not in name)]
#     file = sorted(file)
#     for num, name in enumerate(file):
#         data = loadmat(yourPath + name)
#         points = [x for x in data.keys() if 'DE' in x]
#         output_data[num, :] = data[points[0]][:used_data_len].flatten()
#     return output_data

# load_0hp = Load_hp('0hp')
# load_1hp = Load_hp('1hp')
# load_2hp = Load_hp('2hp')
# load_3hp = Load_hp('3hp')

# def x_y_prepare(load_hp):
#     resample_data_len = data_num*data_len
#     resample_data = np.zeros((class_num, resample_data_len))
#     for i in range(class_num):
#         for j in range(data_num):
#             resample_data[i, j*data_len:(j+1)*data_len] = load_hp[i, j*resample_len:j*resample_len+data_len]    
    
#     load_x = resample_data.reshape([class_num*data_num,data_len])
#     # load_x = np.abs(fft(load_x))[:,0:fft_len]

#     load_y = np.zeros(class_num*data_num)    
#     for i in np.arange(load_y.shape[0]):
#         load_y[i] = i//data_num
    
#     load_x = torch.from_numpy(load_x)
#     load_y = torch.from_numpy(load_y)
#     return load_x, load_y

# def image_prepare(images, labels):
#     images = images.float().to(device)
#     images = torch.unsqueeze(images, 1)
#     labels = labels.long().to(device)
#     return images, labels


data_total_len = 120000
data_len = 500
data_num = 240
fft_len = data_len//2
class_num = 10

# data_len = 120000
# resample_len = 300
# data_len = 1200
# data_num = 400
# used_data_len = resample_len * data_num + data_len
# fft_len = data_len//2
# class_num = 10

def Load_hp(category):
    output_data = np.zeros((10, data_total_len))
    file = [name for name in allFileList if 
            (category in name)&('28'not in name)&('outerrace03'not in name)&('outerrace12'not in name)]
    file = sorted(file)
    # print(file)
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
    load_x = np.abs(fft(load_x))[:,0:fft_len]

    load_y = np.zeros(class_num*data_num)    
    for i in np.arange(load_y.shape[0]):
        load_y[i] = i//data_num
    
    load_x = torch.from_numpy(load_x)
    load_y = torch.from_numpy(load_y)
    return load_x, load_y

def image_prepare(images, labels):
    images = images.to(device)
    labels = labels.to(device)
    # unsqeeze
    images = torch.unsqueeze(images, 1)
    # labels = torch.unsqueeze(labels, 1)
    images = images.float()
    labels = labels.long()
    return images, labels







# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 40
batch_size = 64
source_data_num = 2000
target_data_num = 100

r = source_data_num//target_data_num
ratio = target_data_num//source_data_num
source_x, source_y = x_y_prepare(load_0hp)
target_x, target_y = x_y_prepare(load_3hp)


target_x, target_y = image_prepare(target_x, target_y)
target_x_train, target_x_test, target_y_train, target_y_test = train_test_split(
                                                    target_x, target_y, 
                                                    train_size=target_data_num/10/data_num, 
                                                    test_size=400/10/data_num, 
                                                    shuffle=True)
source_x, source_y = image_prepare(source_x, source_y)
source_x_train, source_x_test, source_y_train, source_y_test = train_test_split(
                                                    source_x, source_y, 
                                                    train_size=source_data_num/10/data_num, 
                                                    test_size=400/10/data_num, 
                                                    shuffle=True)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None
    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)
    
class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.nn1 = nn.Linear(fft_len, 300)         
        self.nn2 = nn.Linear(300, 100)   
        self.leakyrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=51, stride=1, padding=25)
        self.conv2 = nn.Conv1d(5, 10, kernel_size=5, stride=1, padding=2)     
        self.conv3 = nn.Conv1d(10, 15, kernel_size=11, stride=1, padding=5)   
        self.bn1 = nn.BatchNorm1d(5)
        self.bn2 = nn.BatchNorm1d(10)
        self.bn3 = nn.BatchNorm1d(15)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # x = self.nn1(x)
        # x = self.leakyrelu(x)
        # x = self.nn2(x)       
        # x = self.leakyrelu(x)
        # self.featuremap = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.pool(x)        
        x = self.conv3(x)
        x = self.bn3 (x)
        x = self.leakyrelu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        self.featuremap = x
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.nn3 = nn.Linear(465, 10)
        self.nn4 = nn.Linear(100, 10)  
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.nn3(x)
        # x = self.leakyrelu(x)
        # x = self.nn4(x)
        x = F.softmax(x,dim=1)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.nn5 = nn.Linear(465, 2)
        self.nn6 = nn.Linear(50, 2)
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        x = self.nn5(x)
        # x = self.leakyrelu(x)
        # x = self.nn6(x)
        x = F.softmax(x,dim=1)
        return x


E = Extractor().to(device)
C = Classifier().to(device)
D = Discriminator().to(device)

BCELoss = nn.BCELoss().cuda()
CE_Loss = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(list(E.parameters())+
                             list(C.parameters())+
                             list(D.parameters()),
                             lr=0.003, momentum=0.9)

# Train the model
total_step = source_x_train.size(0)//batch_size
start = time.time()
total = 0.0
correct = 0.0
train_loss_values = []
train_acc_value = []
test_loss_values = []
test_acc_value = []
best_accuracy = 0

for epoch in range(num_epochs):
    E.train()
    C.train()
    D.train()
    start_steps = epoch * source_x_train.shape[0]
    total_steps = num_epochs * source_x_train.shape[0]
    start_steps_1 = epoch * target_x_train.shape[0]
    total_steps_1 = num_epochs * target_x_train.shape[0]   
    
    for i in range(total_step):
        images_S = source_x_train[i*batch_size: i*batch_size+batch_size,:]
        labels_S = source_y_train[i*batch_size: i*batch_size+batch_size]
        images_T = target_x_train[i*batch_size//r: i*batch_size+batch_size//r]

        
        p = float(i + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-10 * p)) - 1
        # p_1 = float(i + start_steps_1) / total_step
        # constant_1 = 2. / (1. + np.exp(-10 * p_1)) - 1
        
        
        source_labels = Variable(torch.zeros((images_S.size()[0])).type(torch.LongTensor).to(device))
        target_labels = Variable(torch.ones((images_T.size()[0])).type(torch.LongTensor).to(device))
                
        source_feature = E(images_S)
        source_featuremap = E.featuremap.to(device)
        outputs = C(source_feature)
        
        target_feature = E(images_T)
        target_featuremap = E.featuremap.to(device) 
        class_loss = CE_Loss(outputs, labels_S)

        target_preds = D(target_feature, constant)
        source_preds = D(source_feature, constant)
        target_loss = CE_Loss(target_preds, target_labels)
        source_loss = CE_Loss(source_preds, source_labels)
        domain_loss = (target_loss + source_loss)
        
        optimizer.zero_grad()
        loss = class_loss 
        lossd = 1*domain_loss
        
        loss.backward(retain_graph=True)
        lossd.backward()
        optimizer.step()
        
        print ('Epoch[{}/{}], Iteration[{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    pred = E(source_x_train)
    pred = C(pred)
    _ , prediction = torch.max(pred.data, 1)
    training_accuracy = (prediction == source_y_train).sum().item()/source_x_train.size(0)*100
    print ('Epoch[{}/{}] Accuracy: {:.2f} %' .format(epoch+1, num_epochs, training_accuracy))
    train_loss_values.append(loss.item())
    train_acc_value.append(training_accuracy)
    
    pred = E(target_x_train)
    pred = C(pred)
    _ , prediction = torch.max(pred.data, 1)
    testing_accuracy = (prediction == target_y_train).sum().item()/target_x_train.size(0)*100
    test_acc_value.append(testing_accuracy)
    
    if testing_accuracy >= best_accuracy:
        best_accuracy = testing_accuracy
        # torch.save(model, 'best_model.pth')
        torch.save(E.state_dict(), 'best_E_weight.pth')
        torch.save(C.state_dict(), 'best_C_weight.pth')   

# plt.figure(figsize=(10, 3))     
# plt.subplot(1, 2, 1) 
# plt.plot(train_loss_values)
# plt.title("Loss")
# # plt.subplot(1, 2, 2) 
plt.plot(range(num_epochs), train_acc_value, label='Source Train')
plt.plot(range(num_epochs), test_acc_value, label='Target Train')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
# plt.title("Accuracy")
# plt.savefig("final.jpg")
    
end = time.time()
print('training time: {:.3f} s'.format(end-start))

def get_num_parameters(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
print('number of parameters: ', get_num_parameters(E)+get_num_parameters(C)+get_num_parameters(D))



# Test the model
E = Extractor().to(device)
E.load_state_dict(torch.load('best_E_weight.pth'))
C = Classifier().to(device)
C.load_state_dict(torch.load('best_C_weight.pth'))


# model = torch.load('CDA.pth', map_location=torch.device('cpu'))

def evaluate(data, label):
    E.eval()
    C.eval()
    with torch.no_grad():
        correct = 0
        outputs = E(data)
        outputs = C(outputs)
        _ , predicted = torch.max(outputs.data, 1)
        correct = (predicted == label).sum()
        print('{:.2f} %'.format(100 * correct / data.size(0)))     
        print(data.size(0))

print('Source Domain Train Accuracy : ', end = '')
evaluate(source_x_train, source_y_train)
print('Source Domain Test Accuracy : ', end = '')
evaluate(source_x_test, source_y_test)
print('Target Domain Train Accuracy : ', end = '')
evaluate(target_x_train, target_y_train)
print('Target Domain Test Accuracy : ', end = '')
evaluate(target_x_test, target_y_test)


# # following function (plot_with_labels) is for visualization
# def plot_with_labels(lowDWeights, labels, text):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         color = cm.rainbow(int(255 * s / 10))
#         plt.text(x, y, s, backgroundcolor=color, fontsize=0.5)
#     plt.xlim(X.min(), X.max()) 
#     plt.ylim(Y.min(), Y.max()) 
#     plt.title('Visualized fully connected layer')
#     plt.savefig("TSNE_{}.jpg".format(text))


# # Visualization of trained flatten layer (T-SNE)
# tsne = TSNE(perplexity=30, n_components=2, init='pca')
# plot_only = 1000
# E(source_x_train)
# low_dim_embs = tsne.fit_transform(E.featuremap.cpu().detach().numpy()[:plot_only, :])
# labels = source_y_train.cpu().numpy()[:plot_only]
# plot_with_labels(low_dim_embs, labels, 'source')

# E(target_x_train)
# low_dim_embs = tsne.fit_transform(E.featuremap.cpu().detach().numpy()[:plot_only, :])
# labels = target_y_train.cpu().numpy()[:plot_only]
# plot_with_labels(low_dim_embs, labels, 'target')

def plot_with_labels(lowDWeights, labels, text):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        color = cm.rainbow(int(255 * s / 20))
        plt.text(x, y, s, backgroundcolor=color, fontsize=5)
    plt.xlim(X.min(), X.max()) 
    plt.ylim(Y.min(), Y.max()) 
    plt.title('Visualized fully connected layer')
    plt.savefig("TSNE_{}.jpg".format(text))

# tsne = TSNE(perplexity=30, n_components=2, init='pca')
# plot_only = 300
# E(source_x_train)
# source = E.featuremap.cpu().detach().numpy()[:plot_only, :]
# E(target_x_train)
# target = E.featuremap.cpu().detach().numpy()[:plot_only, :]
# CCC = np.vstack((source,target))
# low_dim_embs = tsne.fit_transform(CCC)
# labels_source = source_y_train.cpu().numpy()[:plot_only]
# labels_target = target_y_train.cpu().numpy()[:plot_only]
# YYY = np.hstack((labels_source,labels_target+10))
# plot_with_labels(low_dim_embs, YYY, 'merge')


