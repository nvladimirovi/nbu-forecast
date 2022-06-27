
# Dataset
# Total number of images: 22495.
# 
# Training set size: 16854 images (one fruit or vegetable per image).
# 
# Test set size: 5641 images (one fruit or vegetable per image).
# 
# Number of classes: 33 (fruits and vegetables).
# 
# Image size: 100x100 pixels.
# 
# Training data filename format: [fruit/vegetable name][id].jpg (e.g. Apple Braeburn100.jpg). Many images are also rotated, to help training.
# 
# Testing data filename format: [4 digit id].jpg (e.g. 0001.jpg)
# 
# 

from __future__ import print_function, division
from multiprocessing import freeze_support

from IPython.display import HTML, display, clear_output

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

if __name__ == '__main__':
    freeze_support()

plt.ion()

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")


# from google.colab import drive
# drive.mount("/drive")


data_dir = './content/dataset'


# !unzip "/content/drive/MyDrive/НБУ/Семестър 2/Прогнозиране чрез анализ на данни - II част. Невронни мрежи/Exam 2/dataset.zip" -d "/content/dataset"
# clear_output()


# Move files in correct dirs
if os.path.exists(data_dir + "/train/train") == True:

    original_train_filenames = os.listdir(data_dir + "/train/train")

    for filename in original_train_filenames:
        os.rename(f"{data_dir}/train/train/{filename}", f"{data_dir}/train/{filename}")

    import shutil

    try:
        shutil.rmtree(f'{data_dir}/train/train')
    except:
        pass


TRAIN = 'train'
TEST = 'test'


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets[TRAIN] = Subset(dataset, train_idx)
    datasets[TEST] = Subset(dataset, val_idx)
    return datasets


search_for_images_in = [
  TRAIN,
  TEST
]

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally. 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.ToTensor(),
    ])
}

whole_dataset = datasets.ImageFolder('./content/dataset/train', transform=transforms.Compose([transforms.ToTensor()]))
image_datasets = train_val_dataset(whole_dataset)
print("Total train size: ", len(image_datasets[TRAIN]))
print("Total test size: ", len(image_datasets[TEST]))

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=4
    )
    for x in search_for_images_in
}

dataset_sizes = {x: len(image_datasets[x]) for x in search_for_images_in}

for x in search_for_images_in:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))
    
print("Classes: ")
class_names = image_datasets[TRAIN].dataset.classes
print(image_datasets[TRAIN].dataset.classes)


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

# Get a batch of training data
def preview():
    inputs, classes = next(iter(dataloaders[TRAIN]))
    show_databatch(inputs, classes)

if __name__ == '__main__':
    preview()

def visualize_model(vgg, num_images=6, type_of_data=TEST):
    was_training = vgg.training
    
    # Set model for evaluation
    vgg.train(False)
    vgg.eval() 
    
    images_so_far = 0

    for i, data in enumerate(dataloaders[type_of_data]):
        inputs, labels = data
        size = inputs.size()[0]
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
        
        outputs = vgg(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]
        
        print("Ground truth:")
        show_databatch(inputs.data.cpu(), labels.data.cpu())
        print("Prediction:")
        show_databatch(inputs.data.cpu(), predicted_labels)
        
        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()
        
        images_so_far += size
        if images_so_far >= num_images:
            break
        
    vgg.train(mode=was_training) # Revert model back to original training state


def eval_model(vgg, criterion, type_of_data=TEST):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    test_batches = len(dataloaders[type_of_data])
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(dataloaders[type_of_data]):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # loss_test += loss.data[0]
        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
        
    avg_loss = loss_test / dataset_sizes[type_of_data]
    avg_acc = acc_test / dataset_sizes[type_of_data]
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


# ## Model creation
# 
# The [VGG-16](https://www.quora.com/What-is-the-VGG-neural-network) is able to classify 1000 different labels; we just need 4 instead. 
# In order to do that we are going replace the last fully connected layer of the model with a new one with 4 output features instead of 1000. 
# 
# In PyTorch, we can access the VGG-16 classifier with `model.classifier`, which is an 6-layer array. We will replace the last entry.
# 
# We can also disable training for the convolutional layers setting `requre_grad = False`, as we will only train the fully connected classifier.


# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=True)

# !!!! IMPORTANT !!!

# ...OR YOU CAN DOWNLOAD THE WEIGHTS FILE ON YOUR OWN AND UNCOMMENT THE CODE BELOW
# IT WOULD BE MUCH FASTER
# vgg16 = models.vgg16_bn()
# vgg16.load_state_dict(torch.load(f"./vgg16_bn.pth/vgg16_bn.pth"))


print(vgg16.classifier[6].out_features) # 1000 


# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
print(vgg16)


list(vgg16.classifier.children())


# If you want to train the model for more than 2 epochs, set this to True after the first run
resume_training = True

if resume_training:
    print("Loading pretrained model..")
    vgg16.load_state_dict(torch.load('VGG16_v2-Fruit-Classifier.pt'))
    print("Loaded!")


if use_gpu:
    vgg16.cuda() #.cuda() will move everything to the GPU side
    
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


print("Test before training")
# eval_model(vgg16, criterion, TEST)


if __name__ == '__main__':
    visualize_model(vgg16) #test before training


def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(dataloaders[TRAIN])
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
        
        for i, data in enumerate(dataloaders[TRAIN]):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break
                
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # loss_train += loss.data[0]
            loss_train += loss.data
            acc_train += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train * 2 / dataset_sizes[TRAIN]
        avg_acc = acc_train * 2 / dataset_sizes[TRAIN]
        
        vgg.train(False)
        vgg.eval()
        
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
        
        best_model_wts = copy.deepcopy(vgg.state_dict())
        # if avg_acc_val > best_acc:
        #     best_acc = avg_acc_val
        #     best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    return vgg


# vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2) # Uncomment to train the model
# torch.save(vgg16.state_dict(), 'VGG16_v2-Fruit-Classifier.pt') # Uncomment to train the model


vgg16.load_state_dict(torch.load(f"VGG16_v2-Fruit-Classifier.pt"))


vgg16.training


# eval_model(vgg16, criterion, TEST) # Uncomment to evaluate the model

if __name__ == '__main__':
    visualize_model(vgg16, num_images=1, type_of_data=TEST)