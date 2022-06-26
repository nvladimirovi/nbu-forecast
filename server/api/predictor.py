
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

from PIL import Image

from server.settings import BASE_DIR

class_names = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine', 'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon']

class Predictor:
    def __init__(self, path_to_model=f'{BASE_DIR}/VGG16_v2-Fruit-Classifier.pt'):
        self.path_to_model = path_to_model

        self.vgg16 = models.vgg16_bn()
        self.vgg16.load_state_dict(torch.load(f"{BASE_DIR}/content/vgg16_bn.pth/vgg16_bn.pth"))
        print(self.vgg16.classifier[6].out_features) # 1000

        # Freeze training for all layers
        for param in self.vgg16.features.parameters():
            param.require_grad = False

        # Newly created modules have require_grad=True by default
        num_features = self.vgg16.classifier[6].in_features
        features = list(self.vgg16.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
        self.vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
        print(self.vgg16)

        print("Loading pretrained model..")
        self.vgg16.load_state_dict(torch.load(self.path_to_model))
        print("Loaded!")

        self.vgg16.train(False)
        self.vgg16.eval() 

    def transform_image(self, image_bytes):
        """
        Transform image into required DenseNet format: 224x224 with 3 RGB channels and normalized.
        Return the corresponding tensor.
        """
        my_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = Image.open(image_bytes)
        return my_transforms(image).unsqueeze(0)