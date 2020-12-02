#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# ## Architecture

# In[2]:

def foobar(x):
    # isnan = torch.isnan(x)
    x[x.isnan()] = 0
    # if torch.any(isnan):
    #     print('stop right there criminal scum')

in_dim = 784  # size of MNIST picture flattened (28x28)
hid_dim = 32  # heuristic I found online
out_dim = 10  # num classes


# ## Data

# In[3]:


import torchvision.datasets as dsets
import torchvision.transforms as transforms
# MNIST Dataset (Images and Labels)

batch_size = 100
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1, 28*28))])
# target_transform = transforms.Lambda(lambda x: F.one_hot(torch.tensor(x), 10))  # num classes

train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=data_transform,
#                             target_transform=target_transform,
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=data_transform)
#                            target_transform=target_transform)

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# ## Utils

# In[4]:


def sigmoid(s):
    return 1 / (1 + torch.exp(-s))

def sigmoid_prime(s):
    # derivative of sigmoid
    # s: sigmoid x
    return s * (1 - s)

def tanh(t):
    return torch.div(torch.exp(t) - torch.exp(-t), torch.exp(t) + torch.exp(-t))

def tanh_prime(t):
    # derivative of tanh
    # t: tanh x
    return 1 - t*t

def softmax(x):
    # x (batch, num_classes)
    # only operates on last dimension
    return x.exp() / x.exp().sum(-1, keepdim=True)

def ce_loss(x, y):  
    # x (batch, num_classes) - network outputs
    # y (batch, 1) - true classes (indexes, NOT one-hot)
    
    loss = 0
    assert x.shape[0] == y.shape[0] # make sure batch sizes are the same
    nll = - torch.log(softmax(x))
    for prob, true_class in zip(nll, y):
        loss += prob[true_class]
    return loss / x.shape[0]

def ce_loss_prime(x, y):
    # x (batch, num_classes) - network outputs
    # y (batch, 1) - true classes (indexes, NOT one-hot)
    
    return softmax(x) - F.one_hot(y, x.size(1)).squeeze()

def eval_model(model, dataloader):
    if isinstance(model, torch.nn.Module):
        model.eval()
    loss, accuracy = 0, 0
    for batch_idx, (x, y) in enumerate(dataloader):
        out = model(x.view(-1,28*28))
        accuracy += accuracy_score(y, out.argmax(1))
        loss += F.cross_entropy(out, y)
    return loss.item() / len(dataloader), accuracy / len(dataloader)


# ## Model

# In[5]:


class Model:
    def __init__(self, in_dim=784, hid_dim=32, out_dim=10, activation="tanh"):
        # architecture
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # parameters
        self.W1 = torch.randn(self.in_dim, self.hid_dim, device=self.device, requires_grad=False)
        self.b1 = torch.ones(self.hid_dim, device=self.device, requires_grad=False)
        self.W2 = torch.randn(self.hid_dim, self.out_dim, device=self.device, requires_grad=False)
        self.b2 = torch.ones(self.out_dim, device=self.device, requires_grad=False)
        
        # activations
        if activation == "tanh":
            self.activation = tanh
            self.activation_prime = tanh_prime
        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        if activation == "relu":
            # self.activation = relu
            # self.activation_prime = relu_prime

    def forward(self, X):
        foobar(X)
        X = X.to(self.device)
        foobar(X)
                                                          # (batch, in_dim)
        Z1 = torch.matmul(X, self.W1) + self.b1           # (batch, hid_dim)
        foobar(Z1)
        self.h = self.activation(Z1)                      # (batch, hid_dim)
        foobar(self.h)
        Z2 = torch.matmul(self.h, self.W2) + self.b2      # (batch, out_dim)
        foobar(Z2)
        return self.activation(Z2)                        # (batch, out_dim)
        
    def backward(self, X, y, out, lr=1e-3):
        X = X.to(self.device)
        y = y.to(self.device)
        out = out.to(self.device)
        batch_size = X.size(0)
        
        dl_dout = (1 / batch_size) * ce_loss_prime(out, y)  # (batch, out_dim)
        foobar(dl_dout)
        dl_dZ2 = dl_dout * self.activation_prime(dl_dout)   # (batch, out_dim)
        foobar(dl_dZ2)
        dl_dW2 = torch.matmul(self.h.t(), dl_dZ2)           # (hid_dim, out_dim)
        foobar(dl_dW2)
        dl_db2 = torch.matmul(torch.ones(batch_size), dl_dZ2)        # (out_dim)
        foobar(dl_db2)
        dl_dH = torch.matmul(dl_dZ2, self.W2.t())           # (batch, hid_dim)
        foobar(dl_dH)
        dl_dZ1 = dl_dH * self.activation_prime(dl_dH)       # (batch, hid_dim)
        foobar(dl_dZ1)
        dl_dW1 = torch.matmul(X.t(), dl_dZ1)                # (in_dim, hid_dim)
        foobar(dl_dW1)
        dl_db1 = torch.matmul(torch.ones(batch_size), dl_dZ1)       # (hid_dim)
        foobar(dl_db1)
        
        self.W1 -= lr * dl_dW1
        self.b1 -= lr * dl_db1
        self.W2 -= lr * dl_dW2
        self.b2 -= lr * dl_db2
        
    def __call__(self, X):
        return self.forward(X)
    
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)        


# ## Creating models

# In[6]:


pt_model = nn.Sequential(
    nn.Linear(in_features=in_dim, out_features=hid_dim),
    nn.Tanh(),
    nn.Linear(in_features=hid_dim, out_features=out_dim),
    nn.Tanh()
)


# In[7]:


model = Model(
    in_dim=784,
    hid_dim=32,
    out_dim=10,
    activation="tanh"
)


# ## Evaluating models before training

# In[ ]:


print("pt_model train \t\t loss {} \t accuracy {}".format(*eval_model(pt_model, train_loader)))
print("pt_model test \t\t loss {} \t accuracy {}".format(*eval_model(pt_model, test_loader)))
print("he_model train \t\t loss {} \t accuracy {}".format(*eval_model(model, train_loader)))
print("he_model test \t\t loss {} \t accuracy {}".format(*eval_model(model, test_loader)))


# ## Training

# ### Our model (HADATZ ELDATZ --YOTAMTZ--)

# In[ ]:


train_accuracy_list = []
test_accuracy_list = []

lr = 1e-2
epochs = 10 

for epoch in range(epochs):
    epoch_train_accuracy = 0
    epoch_test_accuracy = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        out = model(x.view(-1,28*28))
        loss = F.cross_entropy(out, y)
        
        with torch.no_grad():
            epoch_train_accuracy += accuracy_score(y, out.argmax(1))
        model.backward(x.view(-1,28*28), y, out, lr)
        if batch_idx%100 == 0:
            print(f'batch_idx: {int(batch_idx/100)}\tloss: {loss}')
            
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            out = model(x.view(-1,28*28))
            epoch_test_accuracy += accuracy_score(y, out.argmax(1))
    train_accuracy = epoch_train_accuracy / len(train_loader)
    test_accuracy = epoch_test_accuracy / len(test_loader)
    print(f">>>>> epoch: {epoch}\t train accuracy: {train_accuracy}")
    print(f">>>>> epoch: {epoch}\t test accuracy: {test_accuracy}")
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)


# ### Their model (MARK ZUCKERBERG'S MODEL)

# In[ ]:


train_accuracy_list = []
test_accuracy_list = []

pt_model.train()
optimizer = torch.optim.SGD(pt_model.parameters(), lr=lr)

for epoch in range(epochs):
    epoch_train_accuracy = 0
    epoch_test_accuracy = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        out = pt_model(x.view(-1,28*28))
        with torch.no_grad():
            epoch_train_accuracy += accuracy_score(y, out.argmax(1))
        loss = torch.nn.functional.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx%100==0:
            print(f'batch_idx: {int(batch_idx/100)}\tloss: {loss}')
            
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            out = pt_model(x.view(-1,28*28))
            epoch_test_accuracy += accuracy_score(y, out.argmax(1))
    train_accuracy = epoch_train_accuracy / len(train_loader)
    test_accuracy = epoch_test_accuracy / len(test_loader)
    print(f">>>>> epoch: {epoch}\t train accuracy: {train_accuracy}")
    print(f">>>>> epoch: {epoch}\t test accuracy: {test_accuracy}")
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)


# ## Plotting
# 
# Test and train accuracy curves

# In[ ]:


loss_fig, loss_ax = plt.subplots()

plt.plot(train_accuracy_list, label='Train Accuracy')
plt.plot(test_accuracy_list, label='Test Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


# ## Evaluating models after training

# In[ ]:


print("pt_model train \t\t loss {} \t accuracy {}".format(*eval_model(pt_model, train_loader)))
print("pt_model test \t\t loss {} \t accuracy {}".format(*eval_model(pt_model, test_loader)))
print("he_model train \t\t loss {} \t accuracy {}".format(*eval_model(model, train_loader)))
print("he_model test \t\t loss {} \t accuracy {}".format(*eval_model(model, test_loader)))
