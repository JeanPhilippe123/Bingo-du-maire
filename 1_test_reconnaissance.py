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

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#%%
#Download pictures to train
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/Users/JP/Documents/Bingo/1_test_files', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
  batch_size=batch_size_train, shuffle=True)

#Download pictures to train test
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/Users/JP/Documents/Bingo/1_test_files', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
  batch_size=batch_size_test, shuffle=True)

test_loader_6_2 = torch.utils.data.DataLoader(
  torchvision.datasets.ImageFolder('/Users/JP/Documents/Bingo/test_old_bingo',
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
  batch_size=5, shuffle=True)
# for data, target in test_loader:
#     print(data.shape, target.shape)
# for data, target in test_loader:
#     for img in data[0]:
#         plt.figure()
#         plt.imshow(img)
for data, target in test_loader_6_2:
    for img in data:
        for im in img:
            plt.figure()
            plt.imshow(im)
    # print(data.shape, target.shape)
#%%
#Open examples of files with matlplotlib
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
#%%
#Open examples of files with matlplotlib
examples_6_2 = enumerate(test_loader_6_2)
batch_idx, (example_data_6_2, example_targets_6_2) = next(examples_6_2)
fig = plt.figure()
for i in range(5):
  plt.subplot(1,5,i+1)
  plt.tight_layout()
  plt.imshow(example_data_6_2[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets_6_2[i]))
  plt.xticks([])
  plt.yticks([])
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
test_losses_6_2 = []
test_counter_6_2 = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
#%%
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), '/Users/JP/Documents/Bingo/results/model.pth')
      torch.save(optimizer.state_dict(), '/Users/JP/Documents/Bingo/results/optimizer.pth')
      
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def test_6_2():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader_6_2:
      data = torch.from_numpy(np.expand_dims(data[:,0], axis=1))
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader_6_2.dataset)
  test_losses_6_2.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader_6_2.dataset),
    100. * correct / len(test_loader_6_2.dataset)))
#%%
test_6_2()
test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
  test_6_2()
#%%
#Evaluating the model performance
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')

with torch.no_grad():
  output = network(example_data)

fig = plt.figure()
for i in range(4):
  plt.subplot(2,4,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
#%%
#Evaluating the model performance
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter_6_2, test_losses_6_2, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')

with torch.no_grad():
  example_data_6_2 = torch.from_numpy(np.expand_dims(example_data_6_2[:,0], axis=1))
  output = network(example_data_6_2)

fig = plt.figure()
for i in range(5):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data_6_2[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])

#%%
#Continue the training
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)
network_state_dict = torch.load('/Users/JP/Documents/Bingo/results/model.pth')
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load('/Users/JP/Documents/Bingo/results/optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)
#%%
for i in range(4,6):
  test_counter.append(i*len(train_loader.dataset))
  train(i)
  test()
#%%
#Evaluating the model performance
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')