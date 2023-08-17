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
from sklearn.metrics import confusion_matrix


# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Create a file handler and set the level to DEBUG
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)

# Create a stream handler and set the level to INFO
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Set the logger for the root logger as well
logging.root.addHandler(file_handler)
logging.root.addHandler(stream_handler)

# Create a TensorBoard writer
tensorboard_writer = SummaryWriter()

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    logging.info(f'Loaded config file:\n{config}')

except:
    logging.error('Could not load config file.')


try:
    logging.info(f'loading dataset...')
    df = createCSVFromFolder(config['dataset_path'])
    train_df, val_df = getTrainTestSplit(df,config['train_test_split'])

    train_loader = CLFDataset(train_df, dim = config['image_size'],mode='train')
    train_loader = getDataLoader(train_loader)

    val_loader = CLFDataset(val_df, dim = config['image_size'])
    val_loader = getDataLoader(val_loader)

except Exception as e:
    logging.error('Could not load dataset')
    logging.error(e)


# Set the device (e.g., 'cuda' for GPU or 'cpu' for CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config['device'] = device

# try:
model = getModel(config)
model = model.to(device)
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Specify the optimizer and learning rate
optimizer = optim.SGD(model.parameters(), config['lr'], config['momentum'])
# except Exception as e:
#     logging.error('Model loading failed')
#     logging.error(e)


def updateConfig(info, path):
    with open('latest_config.json','w') as f:
        # print(info)
        json.dump(info, f, indent=4)
    



model_name = str(timeit.timeit())
model_path = os.path.join('weights', model_name)
config['model_path']  = model_path

os.makedirs(model_path, exist_ok=True)

updateConfig(config, model_path)



def save_model(model,epoch):
    path = os.path.join(model_path,str(epoch))+'.h5'
    print(f'saving model at: {path}')
    torch.save(model.state_dict(), path)

    config['best_weight'] = path
    updateConfig(config, model_path)


metric_thresh = config['metric_thresh']

# Training loop
def train_model(model, criterion, optimizer, num_epochs):
    best_acc = 0.0
    

    for epoch in range(num_epochs):
        y, y_ = [], []
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                data_loader = train_loader
                model.train()
            else:
                data_loader = val_loader
                model.eval()

            running_loss = []
            running_corrects = []
            precision = []
            recall = []
            f1 = []



            # Iterate over the data
            for inputs, labels in data_loader:
                inputs = inputs.to(config['device']).float()
                labels = labels.to(config['device'])
                # print(labels)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(outputs)
                    labels = labels.view(-1)

                    loss = criterion(outputs, labels)

                    # Backward pass and optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss.append(loss.item()*inputs.shape[0])
                running_corrects.append(torch.sum(preds == labels.data)/inputs.shape[0])


                y_.extend(preds.to('cpu'))
                y.extend(labels.to('cpu'))



                _f1 = f1_score(labels.to('cpu'), preds.to('cpu') > 0.5, average='macro')
                _precision = precision_score(labels.to('cpu'), preds.to('cpu') > 0.5,  average='macro')
                _recall = recall_score(labels.to('cpu'), preds.to('cpu') > 0.5,  average='macro')

                f1.append(_f1)
                precision.append(_precision)
                recall.append(_recall)


            epoch_loss = sum(running_loss) / len(running_loss)
            epoch_acc = sum(running_corrects) / len(running_corrects)
            epoch_f1 = sum(f1) / len(f1)
            epoch_precision = sum(precision) / len(precision)
            epoch_recall = sum(recall) / len(recall)
                
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f}')
            if (epoch_acc > best_acc ) and (phase =='train'):
                best_acc = epoch_acc
                save_model(model, epoch = epoch)

            # Log metrics to TensorBoard
            tensorboard_writer.add_scalar(f'{phase}_loss', epoch_loss, epoch)
            tensorboard_writer.add_scalar(f'{phase}_accuracy', epoch_acc, epoch)

            tensorboard_writer.add_scalar(f'{phase}_recall', epoch_recall, epoch)

            tensorboard_writer.add_scalar(f'{phase}_precision', epoch_precision, epoch)

            tensorboard_writer.add_scalar(f'{phase}_f1', epoch_f1, epoch)

            cm = confusion_matrix(y, y_)
            print(cm)



    # Close the TensorBoard writer
    tensorboard_writer.close()



# Calculate the size of the datasets
# dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}

# Call the training loop
train_model(model, criterion, optimizer, config["num_epochs"])
