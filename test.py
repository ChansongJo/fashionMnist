import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import BayesianLinear as BLinear, MnistModel

from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm

DEVICE = 'cpu'

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

model.load_state_dict(torch.load('model.pt'))

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





y_vecs = model.predict(x_val)
y_pred = np.argmax(y_vecs, axis=1)
y_true = y_val
cm = confusion_matrix(y_true, y_pred)
# print(cm)

# plt.imshow(cm, cmap = 'ocean')
# plt.colorbar

min_val, max_val = 0, 15

# intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
fig, ax = plt.subplots(figsize=(20,20))
ax.matshow(cm, cmap=plt.cm.Blues)
# ax.matshow(cm, cmap=plt.cm.magma_r)
ax.xaxis.set_ticklabels(classes); ax.yaxis.set_ticklabels(classes);

for i in range(10):
    for j in range(10):
        c = cm[j,i]
        ax.text(i, j, str(c), va='center', ha='center')


plt.xticks(range(10))
plt.yticks(range(10))
plt.suptitle('Confusion matrix',size = 32)
plt.xlabel('True labeling',size = 32)
plt.ylabel('Predicted labeling',size = 32)
plt.rcParams.update({'font.size': 28})

# Display some error results 
# y_vecs = model.predict(x_test)
# y_pred = np.argmax(y_vecs, axis=1)
Y_true = y_val
Y_pred_classes =  y_pred
Y_pred = y_vecs
X_val = x_val
# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 8
    ncols = 8
#     plt.figure(figsize=(90, 90))
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(30, 30))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted  :{}\nTrue  :{}".format(classes[pred_errors[error]],classes[obs_errors[error]]), fontsize=14)
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)