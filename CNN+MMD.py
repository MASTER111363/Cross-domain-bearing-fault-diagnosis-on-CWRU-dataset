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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools

# load all data
yourPath = 'CWRU/12k/full_data/'
allFileList = os.listdir(yourPath)

data_total_len = 120000
data_len = 500
data_num = 240
fft_len = data_len//2
class_num = 10

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
    # load_x = np.abs(fft(load_x))[:,0:fft_len]
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
num_epochs = 300
num_classes = 10
batch_size = 128
source_data_num = 2000
target_data_num = 2000
learning_rate = 0.0001
source_x, source_y = x_y_prepare(load_0hp)
target_x, target_y = x_y_prepare(load_3hp)

target_x, target_y = image_prepare(target_x, target_y)
target_x_train, target_x_test, target_y_train, target_y_test = train_test_split(
    target_x, target_y, train_size=target_data_num/10/data_num, test_size=400/10/data_num, random_state=5, shuffle=True)
source_x, source_y = image_prepare(source_x, source_y)
source_x_train, source_x_test, source_y_train, source_y_test = train_test_split(
    source_x, source_y, train_size=source_data_num/10/data_num, test_size=400/10/data_num, random_state=4, shuffle=True)

### 0:ball_07 ; 1:ball_14 ; 2:ball_21
### 3:innerrace_07 ; 4:innerrace_14 ; 5:innerrace_21 ; 6:normal 
### 7:outerrace_07 ; 8:outerrace_14 ; 9:outerrace_21 
# plt.figure(figsize=(12, 9))   
# plt.subplot(3, 3, 1) 
# plt.bar(range(500), np.squeeze(source_x_train[:1,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[:1].cpu()))
# plt.subplot(3, 3, 2) 
# plt.bar(range(500), np.squeeze(source_x_train[1:2,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[1:2].cpu()))
# plt.subplot(3, 3, 3) 
# plt.bar(range(500), np.squeeze(source_x_train[2:3,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[2:3].cpu()))
# plt.subplot(3, 3, 4) 
# plt.bar(range(500), np.squeeze(source_x_train[3:4,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[3:4].cpu()))
# plt.subplot(3, 3, 5) 
# plt.bar(range(500), np.squeeze(source_x_train[4:5,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[4:5].cpu()))
# plt.subplot(3, 3, 6) 
# plt.bar(range(500), np.squeeze(source_x_train[5:6,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[5:6].cpu()))
# plt.subplot(3, 3, 7) 
# plt.bar(range(500), np.squeeze(source_x_train[6:7,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[6:7].cpu()))
# plt.subplot(3, 3, 8) 
# plt.bar(range(500), np.squeeze(source_x_train[7:8,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[7:8].cpu()))
# plt.subplot(3, 3, 9) 
# plt.bar(range(500), np.squeeze(source_x_train[8:9,:].cpu()),width=0.5)
# plt.title(np.array(source_y_train[8:9].cpu()))
# plt.savefig("source3213.jpg")

# plt.figure(figsize=(12, 9)) 
# plt.subplot(3, 3, 1) 
# plt.bar(range(500), np.squeeze(target_x_train[:1,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[:1].cpu()))
# plt.subplot(3, 3, 2) 
# plt.bar(range(500), np.squeeze(target_x_train[1:2,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[1:2].cpu()))
# plt.subplot(3, 3, 3) 
# plt.bar(range(500), np.squeeze(target_x_train[2:3,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[2:3].cpu()))
# plt.subplot(3, 3, 4) 
# plt.bar(range(500), np.squeeze(target_x_train[3:4,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[3:4].cpu()))
# plt.subplot(3, 3, 5) 
# plt.bar(range(500), np.squeeze(target_x_train[4:5,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[4:5].cpu()))
# plt.subplot(3, 3, 6) 
# plt.bar(range(500), np.squeeze(target_x_train[5:6,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[5:6].cpu()))
# plt.subplot(3, 3, 7) 
# plt.bar(range(500), np.squeeze(target_x_train[6:7,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[6:7].cpu()))
# plt.subplot(3, 3, 8) 
# plt.bar(range(500), np.squeeze(target_x_train[7:8,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[7:8].cpu()))
# plt.subplot(3, 3, 9) 
# plt.bar(range(500), np.squeeze(target_x_train[8:9,:].cpu()),width=0.5)
# plt.title(np.array(target_y_train[8:9].cpu()))
# plt.savefig("target3213.jpg")


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=64, stride=1, padding=32)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=11, stride=1, padding=5)     
        self.conv3 = nn.Conv1d(20, 40, kernel_size=5, stride=1, padding=2)   
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(40)   
        self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2480, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
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
        x = self.fc1(x)       
        x = self.leakyrelu(x)
        self.featuremap = x
        x = self.fc2(x)
        x = F.softmax(x,dim=1)
        return x

model = CNN().to(device)


# Loss and optimizer
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0] + target.size()[0])//2
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

CE_Loss = nn.CrossEntropyLoss()
MMD = MMD_loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
total_step = source_x_train.size(0)//batch_size

start = time.time()
total = 0.0
correct = 0.0
running_loss = 0.0
loss_values = []
acc_value = []

for epoch in range(num_epochs):
    model.train()
       
    for i in range(total_step):
        mmd_num = 128
        random_num = random.randrange(target_x_train.size(0)-mmd_num)

        model(target_x_train[random_num:random_num + mmd_num ,:])
        target_feature = model.featuremap.to(device) 
        
        images = source_x_train[i*batch_size:+i*batch_size + batch_size,:]
        labels = source_y_train[i*batch_size:+i*batch_size + batch_size]

        outputs = model(images)
        source_feature = model.featuremap.to(device)
        
        loss = CE_Loss(outputs, labels) + MMD(source_feature, target_feature)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print ('Epoch[{}/{}], Iteration[{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    pred = model(source_x_train)
    _ , prediction = torch.max(pred.data, 1)
    running_accuracy = (prediction == source_y_train).sum().item()/source_x_train.size(0)*100
    print ('Epoch[{}/{}] Accuracy: {:.2f} %' .format(epoch+1, num_epochs, running_accuracy))
    running_loss =+ loss.item()
    loss_values.append(running_loss)
    acc_value.append(running_accuracy)
    
# plt.figure(figsize=(10, 3))     
# plt.subplot(1, 2, 1) 
# plt.plot(loss_values)
# plt.title("Loss")
# plt.subplot(1, 2, 2) 
# plt.plot(acc_value)
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
print('number of parameters: ', get_num_parameters(model))


# Save the model checkpoint
torch.save(model, 'CDA.pth')
torch.save(model.state_dict(), 'CDA_weight.pth')


# Test the model
model = CNN().to(device)
model.load_state_dict(torch.load('CDA_weight.pth'))
# model = torch.load('CDA.pth', map_location=torch.device('cpu'))
model.eval()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def evaluate(data, label):
    model.eval()
    with torch.no_grad():
        correct = 0
        outputs = model(data)
        _ , predicted = torch.max(outputs.data, 1)
        correct = (predicted == label).sum()
        print('{:.2f} %'.format(100 * correct / data.size(0)))     
        print(data.size(0))
        
        # target_names = ['0','1','2','3','4','5','6','7','8','9']
        # month = len(target_names)
        # # print ("month = " + str(month))
        # # print(classification_report(label.cpu(), predicted.cpu(), target_names=target_names))
        # plt.figure()
        # cnf_matrix = confusion_matrix(label.cpu(), predicted.cpu())
        # plot_confusion_matrix(cnf_matrix, classes=target_names,normalize=True,
        #                     title = str(month) + ' confusion matrix')
        # plt.show()


print('Source Domain Train Accuracy : ', end = '')
evaluate(source_x_train, source_y_train)
print('Source Domain Test Accuracy : ', end = '')
evaluate(source_x_test, source_y_test)
print('Target Domain Train Accuracy : ', end = '')
evaluate(target_x_train, target_y_train)
print('Target Domain Test Accuracy : ', end = '')
evaluate(target_x_test, target_y_test)


def plot_with_labels(lowDWeights, labels, text):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        color = cm.rainbow(int(255 * s / 20))
        plt.text(x, y, s, backgroundcolor=color, fontsize=2)
    plt.xlim(X.min(), X.max()) 
    plt.ylim(Y.min(), Y.max()) 
    plt.title('Visualized fully connected layer')
    plt.savefig("TSNE_{}.jpg".format(text))

# tsne = TSNE(perplexity=30, n_components=2, init='pca')
# plot_only = 300
# model(source_x_train)
# source = model.featuremap.cpu().detach().numpy()[:plot_only, :]
# model(target_x_train)
# target = model.featuremap.cpu().detach().numpy()[:plot_only, :]
# CCC = np.vstack((source,target))
# low_dim_embs = tsne.fit_transform(CCC)
# labels_source = source_y_train.cpu().numpy()[:plot_only]
# labels_target = target_y_train.cpu().numpy()[:plot_only]
# YYY = np.hstack((labels_source,labels_target+10))
# plot_with_labels(low_dim_embs, YYY, 'merge')




