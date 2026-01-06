"""
TODO: Finish and submit your code for logistic regression, neural network, and hyperparameter search.

"""

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import math
from tqdm import tqdm


from torch.utils.data import random_split

n_epochs = 19
batch_size_train = 200
batch_size_test = 1000
log_interval = 100
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)



MNIST_training = torchvision.datasets.MNIST('/MNIST_dataset/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

MNIST_test_set = torchvision.datasets.MNIST('/MNIST_dataset/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))


MNIST_training_set, MNIST_validation_set = random_split(MNIST_training, [48000, 12000])
train_loader = torch.utils.data.DataLoader(MNIST_training_set,batch_size=batch_size_train, shuffle=True)
validation_loader = torch.utils.data.DataLoader(MNIST_validation_set,batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(MNIST_test_set,batch_size=batch_size_test, shuffle=True)

CIFAR_training = torchvision.datasets.CIFAR10('.', train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

CIFAR_test_set = torchvision.datasets.CIFAR10('.', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

CIFAR_train_set, CIFAR_val_set = random_split(CIFAR_training, [40000, 10000])
train_loader_fnn = torch.utils.data.DataLoader(CIFAR_train_set, batch_size=batch_size_train, shuffle=True)
val_loader_fnn = torch.utils.data.DataLoader(CIFAR_val_set, batch_size=batch_size_train, shuffle= False)
test_loader_fnn = torch.utils.data.DataLoader(CIFAR_val_set, batch_size=batch_size_test, shuffle= False)


class LogisticRegression(nn.Module):
        def __init__(self):
            super(LogisticRegression,self).__init__()
            self.fc = nn.Linear(28*28,10)

        def forward(self,x):
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            return x

def logistic_regression(device):
    

    def train(epoch,data_loader,model,optimizer):
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output,target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    
    def eval(data_loader,model,dataset):
        loss = 0
        correct = 0
        with torch.no_grad(): 
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                
        loss /= len(data_loader.dataset)
        print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
  

    logistic_regression_model = LogisticRegression().to(device)
    optimizer = optim.SGD(logistic_regression_model.parameters(),lr = 0.015, weight_decay=0.01)

    eval(validation_loader,logistic_regression_model,"Validation")
    for epoch in range(1, n_epochs + 1):
        train(epoch,train_loader,logistic_regression_model,optimizer)
        eval(validation_loader,logistic_regression_model,"Validation")
    
    eval(test_loader,logistic_regression_model,"Test")


    results = dict(
        model= logistic_regression_model,
    )

    return results


class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()

        self.loss_type = loss_type
        self.num_classes = num_classes
        self.fc1 = nn.Linear(32*32*3, 64)  
        self.fc2 = nn.Linear(64, 32)     
        self.fc3 = nn.Linear(32, num_classes)  

    def forward(self, x):
        x = x.view(x.size(0), 32*32*3)  
        x = torch.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        if self.loss_type =='ce':
          output = F.softmax(output, dim=1)  
        
        return output

    def get_loss(self, output, target):
      if self.loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss() 
        loss = loss_fn(output, target)
        return loss

def tune_hyper_parameter(target_metric, device):
    # TODO: implement logistic regression and FNN hyper-parameter tuning here
    def train(epoch, data_loader, model, optimizer):
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output,target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    
    def eval(data_loader,model,dataset):
          loss = 0
          correct = 0
          with torch.no_grad(): 
              for data, target in data_loader:
                  data = data.to(device)
                  target = target.to(device)
                  output = model(data)
                  loss += F.cross_entropy(output, target, reduction='sum').item()
                  pred = output.data.max(1, keepdim=True)[1]
                  correct += pred.eq(target.data.view_as(pred)).sum()  
          loss /= len(data_loader.dataset)
          print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
          return (100. * correct / len(data_loader.dataset))


    def train_fnn(net, optimizer, train_loader, device):
        net.train()
        pbar = tqdm(train_loader, ncols=100, position=0, leave=True)
        avg_loss = 0
        for batch_idx, (data, target) in enumerate(pbar):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            loss = net.get_loss(output, target)
            loss.backward()
            optimizer.step()

            loss_sc = loss.item()

            avg_loss += (loss_sc - avg_loss) / (batch_idx + 1)

            pbar.set_description('train loss: {:.6f} avg loss: {:.6f}'.format(loss_sc, avg_loss))

    def validation(net, validation_loader, device):
        net.eval()
        validation_loss = 0
        correct = 0
        for data, target in validation_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            loss = net.get_loss(output, target)
            validation_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        validation_loss /= len(validation_loader.dataset)
        accuracy = 100. * correct / len(validation_loader.dataset)
        
        print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            validation_loss, correct, len(validation_loader.dataset),
            100. * correct / len(validation_loader.dataset)))
        return accuracy

    best_params = [
        {
            "logistic_regression": {
                "lr": None,
                "wd": None,
            }
        },
        {
            "FNN": {
                "lr" : None,
                "layer" : None,
            }
        }
    ]
    
    best_metric = [
        {
            "logistic_regression": {
                "accuracy": None
            }
        },
        {
            "FNN": {
                "accuracy": None
            }
        }
    ]

    lr_val = [0.001,0.00075,0.00001] 
    wd_val = [0.0001,0.00001] 
    units = [64,128]
    best_metric_lr = 0
    best_metric_fnn = 0
    for lr in lr_val:
        for wd in wd_val:
            logistic_regression_model = LogisticRegression().to(device)
            optimizer = optim.Adam(logistic_regression_model.parameters(), lr=lr, weight_decay=wd)
            for epoch in range(1,9):
                train(epoch, train_loader,logistic_regression_model, optimizer)
            
            accuracy = eval(validation_loader,logistic_regression_model,"Validation")

            if accuracy > best_metric_lr :
                best_metric_lr = accuracy
                best_params[0]["logistic_regression"]["lr"] = lr
                best_params[0]["logistic_regression"]["wd"] = wd
                best_metric[0]["logistic_regression"]["accuracy"] = accuracy
    
    for lr in lr_val:
        for unit in units:
            fnn_model = FNN("ce",10).to(device)
            optimizer_fnn = optim.Adam(fnn_model.parameters(), lr=lr) 
            
            for epoch in range(1,9):
                train_fnn(fnn_model,optimizer_fnn, train_loader_fnn, device)
            
            accuracy = validation(fnn_model,val_loader_fnn,device)
            if accuracy > best_metric_fnn:
                best_metric_fnn = accuracy
                best_params[1]["FNN"]["lr"] = lr
                best_params[1]["FNN"]["layer"] = unit
                best_metric[1]["FNN"]["accuracy"] = accuracy
    
    

    return best_params, best_metric