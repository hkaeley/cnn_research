#Basic framework created with support from https://github.com/madsendennis/notebooks/tree/master/pytorchytorch
torch, torchvision
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


#these settings are required to be used by the pytorch models

def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    
    history = []
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


def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image
  
    '''
      
    transform = image_transforms['test']
  
    test_image = Image.open(test_image_name)
#     plt.imshow(test_image)
      
    test_image_tensor = transform(test_image)
  
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
      
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)
        for i in range(3):
            print("Prediction", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])

def prediction_accuracy(test_data_location, model):
    total = 0
    correct = 0
    test_data_locations = [x for x in Path(test_data_location).iterdir()]
#     import pdb; pdb.set_trace()
    for image_loc in test_data_locations:
        try:
            transform = image_transforms['test']
        
            test_image = Image.open(image_loc)
        #     plt.imshow(test_image)
            
            test_image_tensor = transform(test_image)
        
            if torch.cuda.is_available():
                test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
            else:
                test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
            
            with torch.no_grad():
                model.eval()
                # Model outputs log probabilities
                out = model(test_image_tensor)
                ps = torch.exp(out)
                topk, topclass = ps.topk(3, dim=1)
                total += 1
                
                if idx_to_class[topclass.cpu().numpy()[0][0]] == str(image_loc.stem.split('_')[0]):
                    correct+=1
        except:
            print(f"failed prediction for: {image_loc}")
    print(f"{(correct/total)*100}% correct for testing")
            



image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), #introduce more variety
        transforms.RandomRotation(degrees=15), #introduce more variety
        transforms.RandomHorizontalFlip(), #add possibility of horizontal fit to increase variety
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256), #in valid and test you should only normalize the images (not extra randomization needed)
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
}


def return_model(model_choice):
    if model_choice == "alexnet":
        model = models.alexnet(pretrained=True)
        modules = [module for module in model.modules() if not isinstance(module,nn.Sequential)]
        modules.remove(model)
        modules_limit = len(modules) - 4 #we want to only unfreeeze last 5 layers, only 4 because we will add a layer at the end
        count = 0
        for module in modules:
            if count < modules_limit:
                for param in module.parameters():
                    param.requires_grad = False
                count += 1
         
            else:
                break;
        model.classifier[6] = nn.Linear(4096, num_classes) #change the final layer to fit our classes since the normal output would be for 1000 classes 
        model.classifier.add_module("7", nn.LogSoftmax(dim = 1)) #softmax takes input and normalizes the probabilites 
        model.to(device) #making compatible with gpu usage
        return model

    elif model_choice == "resnet18":
        model = models.resnet18(pretrained=True)
        modules = [module for module in model.modules() if not isinstance(module,nn.Sequential)]
        modules.remove(model)
        modules_limit = len(modules) - 4 #we want to only unfreeeze last 5 layers, only 4 because we will add a layer at the end
        count = 0
        for module in modules:
            if count < modules_limit:
                for param in module.parameters():
                    param.requires_grad = False
                count += 1
          
            else:
                break;
        model.fc = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim = 1)
        )
        model.to(device) #making compatible with gpu usage
        return model

    elif model_choice == "vgg16":
        model = models.vgg16(pretrained=True)
        for param in model.parameters(): #basically use only the last layer for backpropogation since we want to edit the alexnet model below
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes) #change the final layer to fit our classes since the normal output would be for 1000 classes 
        model.classifier.add_module("7", nn.LogSoftmax(dim = 1)) #softmax takes input and normalizes the probabilites 
        model.to(device) #making compatible with gpu usage
        return model



# Load the Data

# Set train and valid directory paths

# dataset = r"C:\Users\harsi\research\notebooks\pytorch\caltec256subset"
dataset = r"C:\Users\harsi\research\output"
train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'valid')
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

device = torch.device("cuda")
  
model = return_model("resnet18")
loss_func = nn.NLLLoss()                    
optimizer = optim.Adam(model.parameters())



num_epochs = 22
# trained_model, history = train_and_validate(alexnet, loss_func, optimizer, num_epochs)
# trained_model, history = train_and_validate(resnet18, loss_func, optimizer, num_epochs)
trained_model, history = train_and_validate(model, loss_func, optimizer, num_epochs)

torch.save(history, dataset+'_history.pt')

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

prediction_accuracy(test_directory, model)


# predict(trained_model, dataset + r'\test\bear_1.jpg')
# predict(trained_model, dataset + r'\test\gorilla_1.jpg')
# predict(trained_model, dataset + r'\test\other_1.jpg')

# predict(trained_model, dataset + r'\test\Diana, Princess of Wales_0.jpg')
# predict(trained_model, dataset + r'\test\Amelia Earhart_0.jpg')
# predict(trained_model, dataset + r'\test\Nancy Pelosi_0.jpg')


















