
import torch
import torch.nn as nn
import torch.optim as optim
from model import getModel
import json
from datasets import CLFDataset, getDataLoader, createCSVFromFolder, getTrainTestSplit
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import timeit
import json
from tqdm import tqdm
import itertools
import numpy as np



try:
    with open(os.path.join('latest_config.json'), 'r') as f:
        config = json.load(f)
    logging.info(f'Loaded config file:\n{config}')

except:
    logging.error('Could not load config file.')


df = createCSVFromFolder(config['validation_dataset_path'], eval = True)
valid, _ = getTrainTestSplit(df,0.01)

valid_loader = CLFDataset(valid,)
valid_loader = getDataLoader(valid_loader)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

running_loss = []
running_corrects = []
precision = []
recall = []
f1 = []

model = getModel(config)
model = model.to(device)
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Specify the optimizer and learning rate
optimizer = optim.SGD(model.parameters(), config['lr'], config['momentum'])

best_weight = config['best_weight']

model.load_state_dict(torch.load(os.path.join('./',best_weight)))
model.eval()

y = []
y_ = []

# Iterate over the data
for inputs, labels in tqdm(valid_loader):
    inputs = inputs.to(device).float()
    labels = labels.to(device)

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        labels = labels.view(-1)

        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs,1).to('cpu')
        # print(probs, labels)


    # Statistics
    running_loss.append(loss.item()*inputs.shape[0])
    running_corrects.append(torch.sum(preds == labels.data)/inputs.shape[0])

    y_.extend(preds.to('cpu'))
    y.extend(labels.to('cpu'))

    # _f1 = f1_score(labels.to('cpu'), probs > 0.5, average='micro')
    # _precision = precision_score(labels.to('cpu'),probs > 0.5 > 0.5,  average='micro')
    # _recall = recall_score(labels.to('cpu'), probs > 0.5.to('cpu') > 0.5,  average='micro')
    # # print(_f1, preds.to('cpu'),labels.to('cpu') )
    # f1.append(_f1)
    # precision.append(_precision)
    # recall.append(_recall)


epoch_loss = sum(running_loss) / len(running_loss)
epoch_acc = sum(running_corrects) / len(running_corrects)
# epoch_f1 = sum(f1) / len(f1)
# epoch_precision = sum(precision) / len(precision)
# epoch_recall = sum(recall) / len(recall)

# print(f'test >>> F1: {epoch_f1:.4f} Precision: {epoch_precision:.4f}  Recall: {epoch_recall:.4f}')
print(f'test >>> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')




# confusion metric on class level

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_)

with open('categories.json','r') as f:
    cats = json.load(f)

cat_names = list(cats.keys())
plot_confusion_matrix(cm, cat_names)


