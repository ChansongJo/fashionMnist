import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import BayesianLinear as BLinear, MnistModel

from tqdm import tqdm

DEVICE = 'cuda'

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
batch_size = 200
model = MnistModel(BLinear)
#model = nn.DataParallel(model)
model.to(DEVICE)
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('fashionData', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('fashionData', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size)


optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8, last_epoch=-1)
critetion = nn.CrossEntropyLoss()

i = 0
for epoch in range(45):
    model.train()
    total_loss = 0
    accuracy = 0
    for data, target in tqdm(train_loader):
        data, target = Variable(data).to(DEVICE), Variable(target).to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = critetion(output, target)
        loss.backward()    # calc gradients
        optimizer.step()   # update gradients
        prediction = torch.argmax(output, 1)  
        total_loss += loss.detach()
        accuracy += prediction.eq(target).sum().item() / batch_size

    training_len = len(train_loader)
    curr_lr = scheduler.get_lr()[0]
    print('Epoch: {}\tLoss: {:.3f}\tAccuracy: {:.3f}\tcurr_lr: {:.6f}'.format(epoch, total_loss/training_len, accuracy/training_len, curr_lr))
    if curr_lr > 0.0003:
        scheduler.step()
    
    model.eval()
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data).to(DEVICE), Variable(target).to(DEVICE)
            output = model(data)
            prediction = torch.argmax(output, 1)
            accuracy += prediction.eq(target).sum().item()
    print('Eval Accuracy: {:.4f}'.format(accuracy/len(test_loader.dataset)))


torch.save(model.state_dict(), 'model.pt')
print('Successfully Saved model to model.pt')

'''
model.eval()
accuracy = 0
for data, target in test_loader:
    with torch.no_grad():
        data, target = Variable(data).to(DEVICE), Variable(target).to(DEVICE)
    output = model(data)
    prediction = torch.argmax(output, 1)
    #print(target)
    #print(prediction)
    accuracy += prediction.eq(target).sum().item()

print('\nTest set: Accuracy: {:.4f}'.format(accuracy/len(test_loader.dataset)))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images.to(DEVICE))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.to(DEVICE)).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %% %s out of %s' % (
        classes[i], 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))
'''