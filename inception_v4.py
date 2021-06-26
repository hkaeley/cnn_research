import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
# from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

import pretrainedmodels

#these settings are required to be used by the pytorch models

def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in tqdm(range(epochs)):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):
            
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        # Save if the model has best accuracy till now
        #torch.save(model, dataset+'_model_'+str(epoch)+'.pt')
            
    return model, history


def prediction_accuracy(test_data_location, model):
    total = 0
    correct = 0
    test_data_folder_locations = [x for x in Path(test_data_location).iterdir()]
#     import pdb; pdb.set_trace()
    test_data_locations = []
    for dirs in test_data_folder_locations:
        test_data_locations += [x for x in dirs.iterdir()]
  
    for image_loc in test_data_locations:
        try:
            transform = image_transforms['test']
        
            test_image = Image.open(image_loc).convert("RGB")
        #     plt.imshow(test_image)
            
            test_image_tensor = transform(test_image)
            test_image_tensor = test_image_tensor.view(1, 3, 299, 299)
            
            with torch.no_grad():
                model.eval()
                # Model outputs log probabilities
                
                out = model(test_image_tensor)
                ps = torch.exp(out)
                topk, topclass = ps.topk(2, dim=1)
                total += 1
                
                if idx_to_class[topclass.cpu().numpy()[0][0]] in str(image_loc):
                    correct+=1
        except:
            try:
                transform = image_transforms['test']
            
                test_image = Image.open(image_loc)
            #     plt.imshow(test_image)
                
                test_image_tensor = transform(test_image)
                test_image_tensor = test_image_tensor.view(299, 299, 3)
                
                with torch.no_grad():
                    model.eval()
                    # Model outputs log probabilities
                    out = model(test_image_tensor)
                    ps = torch.exp(out)
                    topk, topclass = ps.topk(3, dim=1)
                    total += 1
                    
                    if idx_to_class[topclass.cpu().numpy()[0][0]] in str(image_loc):
                        correct+=1
            except:
                print(f"failed prediction for: {image_loc}")
    print(f"{(correct/total)*100}% correct for testing")
            
            

image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=400, scale=(0.8, 1.0)), #introduce more variety
        transforms.RandomRotation(degrees=15), #introduce more variety
        transforms.RandomHorizontalFlip(), #add possibility of horizontal fit to increase variety
        transforms.CenterCrop(size=299),
        transforms.ToTensor(),
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=400), #in valid and test you should only normalize the images (not extra randomization needed)
        transforms.CenterCrop(size=299),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=400),
        transforms.CenterCrop(size=299),
        transforms.ToTensor(),
        ])
}

# Load the Data

# Set train and valid directory paths

dataset = r"C:\Users\harsi\research\chest_xray"
train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'val')
test_directory = os.path.join(dataset, 'test')

# Batch size
bs = 1

# Number of classes
num_classes = len(os.listdir(valid_directory))  #1 folder per class

# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
#     'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

# Get a mapping of the indices to the class names, in order to see the output classes of the test images.
idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()} #give indeces to class names

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])

# Create iterators for the Data loaded using DataLoader module

train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)
# test_data_loader = DataLoader(data['test'], batch_size=bs)

device = torch.device("cpu")


incept = pretrainedmodels.inceptionv4(num_classes=2, pretrained=None)


# loss_func = nn.NLLLoss()
loss_func = nn.CrossEntropyLoss()
# optimizer = optim.Adam(incept.parameters())                       
optimizer = torch.optim.SGD(incept.parameters(), 0.1,
                            momentum=0.9,
                            weight_decay=0.0001)


num_epochs = 10
trained_model, history = train_and_validate(incept, loss_func, optimizer, num_epochs)
 
try:
    torch.save(history, dataset+'_history.pt')
except:
    pass
 
history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
# plt.ylim(0,1)
plt.savefig(dataset+'_loss_curve.png')
plt.show()
 
 
plt.plot(history[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
# plt.ylim(0,1)
plt.savefig(dataset+'_accuracy_curve.png')
plt.show()

prediction_accuracy(test_directory, incept)


# predict(trained_model, dataset + r'\test\bear_1.jpg')
# predict(trained_model, dataset + r'\test\gorilla_1.jpg')
# predict(trained_model, dataset + r'\test\other_1.jpg')


# predict(trained_model, dataset + r'\test\Diana, Princess of Wales_0.jpg')
# predict(trained_model, dataset + r'\test\Amelia Earhart_0.jpg')
# predict(trained_model, dataset + r'\test\Nancy Pelosi_0.jpg')


















