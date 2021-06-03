from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F 
import time
import random
import datetime
import os
import pandas as pd 
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import Tensor
import functools
from torch.autograd import Variable
from dataset import current
from torchvision import datasets, transforms
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from multiresnet1d import MSResNet
from myresnet1d import ResNet1d
from TheResNet import TheResNet
from inceptionTime import InceptionTime
from sklearn.metrics import confusion_matrix
import csv


random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic=True



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)
valid_size = 0.3
batch_size = 360
test_bs = 80

# load_file = 'output_woslide_3c_fixed_MAX.csv'
load_file = 'TG_output_woslide_fixed_MAX.csv'
downsampletype = load_file.split('_')[-1].split('.')[0]
print('Loading file...', load_file)
df = pd.read_csv(load_file)
print('File loaded!')
# current = df.as_matrix()
current = df.values
label = np.array(current[:,0])
num_data = len(current)
current = np.array(current[:,1:])
maximum = current.max()
minimum = current.min()
current = (current - (maximum+minimum)/2)/(maximum - (maximum+minimum)/2)
maxi = current.max()
mini = current.min()
print(maxi, mini)

current = torch.from_numpy(current).type(torch.FloatTensor)
label = torch.from_numpy(label).type(torch.LongTensor)
current = current.view(num_data, 1, -1)
label = label.view(num_data, 1)


Train_Set = TensorDataset(current, label)
Valid_Set = TensorDataset(current, label)
Test_Set = TensorDataset(current, label)


num_train = len(Train_Set)
indices = list(range(num_train))
split = int(np.floor(valid_size*num_train))
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
valid_idx, test_idx = valid_idx[:len(valid_idx)//2], valid_idx[len(valid_idx)//2:]
train_idx=train_idx[:len(train_idx)//batch_size*batch_size]
valid_idx=valid_idx[:len(valid_idx)//test_bs*test_bs]
test_idx =test_idx[:len(test_idx)//test_bs*test_bs]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = DataLoader(Train_Set, batch_size=batch_size, sampler= train_sampler,
                        num_workers=4, pin_memory=True)

valid_loader = DataLoader(Valid_Set, batch_size=test_bs, sampler = valid_sampler,
                      num_workers=4, pin_memory=True)

test_loader = DataLoader(Test_Set, batch_size=test_bs, sampler = test_sampler,
                      num_workers=4, pin_memory=True)




def save_model(state,save_model_path,modelname):
    filename = os.path.join(save_model_path,modelname+'.pth')
    torch.save(state,filename)

def load_model(Net, optimizer, model_file):
    assert os.path.exists(model_file),'There is no model file from'+model_file
    checkpoint = torch.load(model_file)
    Net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']+1
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return Net, optimizer, start_epoch

def train(model, data_loader, opt, loss, epoch,verbose = True):
    model.train()
    loss_avg = 0.0
    correct = 0
    for batch_idx, (data,target) in enumerate(data_loader):
        # print('input:',data.shape)
        target = target.squeeze()
        data, target = Variable(data.to(device)), Variable(target.to(device))
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape, target.shape)
        loss = loss_fn(output, target)
        loss_avg = loss_avg + loss.item()
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # if batch_idx == len(data_loader)-1:
            # print(pred)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        verbose_step = len(data_loader) 
        if (batch_idx+1)  % verbose_step == 0 and verbose:
            print('Train Epoch: {}  Loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, loss.item(), correct, len(train_idx),
            100. * correct / (len(train_idx))))
    return loss_avg / (len(train_idx))

def test(model, data_loader, loss, testing=False):
    if testing:
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        confu_target = []
        confu_pred   = []
        incorrect = [[], []]
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in data_loader:
            target = target.squeeze()
            data, target = Variable(data.to(device)), Variable(target.to(device))
            output = model(data)
            test_loss += loss_fn(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            
            if testing:
                confu_target.extend(target.tolist())
                confu_pred.extend(pred.squeeze(1).tolist())   
            
                correct_tensor = pred.eq(target.data.view_as(pred))
                correct_tensor = np.squeeze(correct_tensor.cpu().numpy())
                for i in range(len(target)):       
                    label = target.data[i]
                    class_correct[label] += correct_tensor[i].item()
                    class_total[label] += 1

                    if correct_tensor[i].item() == 0:
                        incorrect[label].append(data.data[i].cpu().tolist())
                if incorrect[0] != []:
                    with open('./incorrect/incorrect0.csv', 'w', newline='') as writefile:
                        writer = csv.writer(writefile)
                        for wrong in incorrect[0]:
                            writer.writerow(wrong[0])
                if incorrect[1] != []:
                    with open('./incorrect/incorrect1.csv', 'w', newline='') as writefile:
                        writer = csv.writer(writefile)
                        for wrong in incorrect[1]:
                            writer.writerow(wrong[0])

        test_loss /= len(data_loader.dataset)
        if testing:
            print('Testing set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_idx),
                100. * correct / len(test_idx)))
        else:
            print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(valid_idx),
                100. * correct / len(valid_idx)))    
            
        if testing:
            for i in range(2):
                if class_total[i] > 0:
                    print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                        i, 100 * class_correct[i] / class_total[i],
                        np.sum(class_correct[i]), np.sum(class_total[i])))
                else:
                    print('Test Accuracy of %5s: N/A (no testing examples)' % (class_total[i]))
            print('\n')
            print(confusion_matrix(confu_target, confu_pred))
            
    return float(correct) / len(valid_idx)

def CNNBest(model_name='TheResNet', pretrain=None):
    if model_name == 'MSResNet':
        net = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=3)
    elif model_name == 'ResNet1d':
        net = ResNet1d(input_channel=1, layers=[1, 1, 1, 1], num_classes=3)
    elif model_name == 'TheResNet':
        # net = TheResNet(n_in=1075, n_classes=3) # Gas type classification
        net = TheResNet(n_in=1018, n_classes=2) # Townsend Glow classification
    elif model_name == 'InceptionTime':
        # net = InceptionTime(time_steps=1075, nb_classes=3)# Gas type classification
        net = InceptionTime(time_steps=1018, nb_classes=2)# Townsend Glow classification
    
    
    # TheResNet with pretrain
    if pretrain:
        net = TheResNet(n_in=1075, n_classes=6)
        checkpoint = torch.load('./pretrain/synthetic_control_0316/best_acc.pth')
        net.load_state_dict(checkpoint['model_state_dict'])
        net.fc1 = nn.Linear(128, 3)
        net.n_classes = 3
    
    return net


model_name = 'InceptionTime'
net = CNNBest(model_name).to(device)

# loss_fn = Focal_Loss()
loss_fn = torch.nn.CrossEntropyLoss()
    
# Train the model
# num_epochs = 2000 # TheResNet
# learning_rate = 0.000001 # TheResNet
num_epochs = 500
learning_rate = 0.0001
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00000001)

# Model weighting path
today = datetime.date.today().strftime('%y%m%d')[2:]
output_path = os.path.join('log', model_name+'_'+today+'_'+downsampletype)
os.makedirs(output_path, exist_ok=True)

resume = None

StartTime = time.time()
loss, val_acc, lr_curve = [], [], []

if resume is not None:
    net, optimizer, start_epoch = load_model(net, optimizer, resume)
    print('load', resume)

best_acc = 0.3
for epoch in range(num_epochs):
    #lr = adjust_learning_rate(learning_rate, optimizer, epoch, epoch_list=[80, 170])
    train_loss = train(net, train_loader, optimizer, loss_fn, epoch, verbose=True)
    valid_acc  = test(net, valid_loader, loss_fn)
    loss.append(train_loss)
    val_acc.append(valid_acc)
    lr_curve.append(learning_rate)
    # lr_curve.append(scheduler.get_last_lr()[0])

    #if (epoch+1)%50 == 0 or epoch==num_epochs-1:
    #    save_model({'epoch':epoch,
    #                'model_state_dict':net.state_dict(),
    #                'optimizer_state_dict':optimizer.state_dict(),
    #                },
    #                os.path.join(output_path,'savemodel'),model_name)

    if valid_acc>=best_acc:
        print(valid_acc)
        save_model({'epoch':epoch,
                    'model_state_dict':net.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    },
                    output_path,
                    'best_acc')
        best_acc = valid_acc


    if (epoch+1)%100 == 0 and epoch <= 1000:
        learning_rate /= 10
    # scheduler.step()


checkpoint = torch.load(os.path.join(output_path,'best_acc.pth'))
net.load_state_dict(checkpoint['model_state_dict'])
print('Best epoch:', checkpoint['epoch'])
test_acc = test(net, test_loader, loss_fn, testing=True)

torch.save(CNNBest, 'CNNx.pkl')
EndTime = time.time()
print('Time Usage: ', str(datetime.timedelta(seconds=int(round(EndTime-StartTime)))))



plt.figure()
plt.plot(loss)
plt.title('Train Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.figure()
plt.plot(val_acc)
plt.title('Valid Acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.figure()
plt.plot(lr_curve)
plt.title('Learning Rate')
plt.xlabel('epochs')
plt.ylabel('lr')
plt.yscale('log')
plt.show()
