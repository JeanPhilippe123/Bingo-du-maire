"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""
import torch
import torchvision
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np

n_epochs = 1
batch_size_train = 200
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#%%
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.ImageFolder('/Users/JP/Documents/Crosser-le-Maire/Cartes_bingo_1/Numbers',
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
  batch_size=180, shuffle=False)
#%%
#Add targets
train_targets = np.array([1,6,4,9,2,0,6,5,1,3,5,2,8,6,3,7,4,0,5,4,6,4,0,9,8,0,2,5,
                    3,3,1,6,5,8,7,5,4,6,6,6,5,0,3,5,3,3,2,2,2,1,6,5,7,6,0,8,
                    1,4,5,3,8,7,3,4,4,4,2,2,7,1,4,3,4,6,6,4,5,7,6,2,4,2,6,9,
                    5,4,8,1,1,1,1,0,5,8,6,4,6,9,3,9,2,1,0,3,3,6,8,4,6,3,1,6,
                    2,1,4,7,3,7,4,8,2,8,4,7,1,8,5,3,2,7,7,6,8,5,1,4,3,9,1,9,
                    8,6,3,5,1,3,0,2,3,5,1,9,6,5,9,7,3,0,3,2,4,2,6,9,4,2,3,8,
                    2,1,3,2,6,5,9,5,4,5,4,1])
train_data = list(train_loader)[0]
train_image = train_data[0]
train_image =  torch.from_numpy(np.expand_dims(train_image[:,0], axis=1))
# for i in range(0,len(train_data[0])):
#     plt.figure()
#     plt.imshow(train_data[0][i][0])
#     plt.title(train_targets[i])
#%%
#Constructing the network
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Class network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)    
#%%
#Training the network
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
#%%
def train(epoch,train_loader,targets):
  network.train()
  for batch_idx, (data, target) in enumerate(list(train_loader)):
    data = torch.from_numpy(np.expand_dims(data[:,0], axis=1))
    optimizer.zero_grad()
    output = network(data)
    tar = torch.from_numpy(targets)
    loss = F.nll_loss(output, tar)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), '/Users/JP/Documents/Crosser-le-Maire/results/model.pth')
      torch.save(optimizer.state_dict(), '/Users/JP/Documents/Crosser-le-Maire/results/optimizer.pth')
      
# def test():
#   network.eval()
#   test_loss = 0
#   correct = 0
#   with torch.no_grad():
#     for data, target in test_loader:
#       output = network(data)
#       test_loss += F.nll_loss(output, target, size_average=False).item()
#       pred = output.data.max(1, keepdim=True)[1]
#       correct += pred.eq(target.data.view_as(pred)).sum()
#   test_loss /= len(test_loader.dataset)
#   test_losses.append(test_loss)
#   print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#     test_loss, correct, len(test_loader.dataset),
#     100. * correct / len(test_loader.dataset)))

# def test_6_2():
#   network.eval()
#   test_loss = 0
#   correct = 0
#   with torch.no_grad():
#     for data, target in test_loader_6_2:
#       data = torch.from_numpy(np.expand_dims(data[:,0], axis=1))
#       output = network(data)
#       test_loss += F.nll_loss(output, target, size_average=False).item()
#       pred = output.data.max(1, keepdim=True)[1]
#       correct += pred.eq(target.data.view_as(pred)).sum()
#   test_loss /= len(test_loader_6_2.dataset)
#   test_losses_6_2.append(test_loss)
#   print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#     test_loss, correct, len(test_loader_6_2.dataset),
#     100. * correct / len(test_loader_6_2.dataset)))
#%%
for epoch in range(1, n_epochs + 1):
  train(epoch,train_loader,train_targets)
#%%
#Evaluating the model performance
with torch.no_grad():
  output = network(train_image)

fig = plt.figure()
for i in range(4):
  plt.subplot(2,4,i+1)
  plt.tight_layout()
  plt.imshow(train_image[i][0], interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
#%%
#Continue the training
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)
network_state_dict = torch.load('/Users/JP/Documents/Crosser-le-Maire/results/model.pth')
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load('/Users/JP/Documents/Crosser-le-Maire/results/optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)
# #%%
# for i in range(4,6):
#   test_counter.append(i*len(train_loader.dataset))
#   train(i)
#   test()
#%%
#Evaluating the model performance
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')