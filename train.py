import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
from PIL import Image
import numpy as np
import seaborn as sb
import json


parser = argparse.ArgumentParser(description='Train a Model.')
parser.add_argument('data_directory')
parser.add_argument('--save_dir')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--arch')
parser.add_argument('--learning_rate')
parser.add_argument('--hidden_units')
parser.add_argument('--epochs')

args = parser.parse_args()

data_dir= args.data_directory

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
testing_transforms= transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
training_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_datasets = datasets.ImageFolder(valid_dir, transform=testing_transforms)
testing_datasets= datasets.ImageFolder(test_dir, transform=testing_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
training_dataloaders = torch.utils.data.DataLoader(training_datasets, batch_size=64, shuffle=True)
validation_dataloaders=torch.utils.data.DataLoader(validation_datasets, batch_size=64)
testing_dataloaders= torch.utils.data.DataLoader(testing_datasets, batch_size=64)




with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    
use_gpu = False
#vgg16 is the default in case the optional argument was not supplied
use_vgg16 =False
use_densenet161=False
if(args.gpu is not None and args.gpu):
    print("we are to use GPU")
    use_gpu= True


if(args.arch is not None):
    print("The chosen arch is  " + args.arch)
    #this can be optimized to be an if else only for sure
    if args.arch.lower() =="vgg16":
        use_vgg16= True
    elif args.arch.lower() == "densenet161":
        use_densenet161= True
    else:
        use_vgg16= True
else:
    print("Arch was not supplied")
    use_vgg16= True
    
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu  ) else "cpu")
if use_densenet161:
    model= models.densenet161(pretrained=True)
else:
    model = models.vgg16(pretrained=True)

#default value for epoch in case it was not supplied as an argument
numberOfEpochs = 5
if(args.epochs is not None):
    print("the epoch value is " + args.epochs)
    numberOfEpochs = int(args.epochs)

else:
    print("epochs were not supplied")
    
learning_rate= 0.001
if(args.learning_rate is not None):
    learning_rate= float(args.learning_rate)
else:
    print("learning_rate was not supplied")

hidden_units= 2048
if(args.hidden_units is not None):
    hidden_units= int(args.hidden_units)
else:
    print("hidden_units was not supplied")
for param in model.parameters():
    param.requires_grad=False
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1',nn.Dropout(0.3)),
                          ('fc2', nn.Linear(hidden_units, 1024)),
                          ('relu2',nn.ReLU()),
                          ('drop2',nn.Dropout(0.3)),
                          ('fc3', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier
model.to(device)
optimizer=optim.Adam(model.classifier.parameters(), lr=learning_rate)
criterion= nn.NLLLoss()

#could have been a separate function for a cleaner code

#starting the training
print("starting the training")
epochs=numberOfEpochs
steps=0
running_loss=0
print_every= 5
for epoch in range(epochs):
    for inputs, labels in training_dataloaders:
        steps+=1
        # Move input and label tensors to the default device
        #print(str(device))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        inputs, labels= inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        #training
        logps= model.forward(inputs)
        loss= criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss+= loss.item()
        print("now starting the validation")
        if steps % print_every == 0:
            validation_loss= 0
            accuracy=0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validation_dataloaders:
                    torch.set_default_tensor_type('torch.cuda.FloatTensor')
                    inputs, labels= inputs.to(device), labels.to(device)
                    logps= model.forward(inputs)
                    batch_loss= criterion(logps, labels)
                    validation_loss+=batch_loss.item()
                    # Calculate accuracy
                    ps= torch.exp(logps)
                    top_p, top_class= ps.topk(1, dim=1)
                    equals= top_class== labels.view(*top_class.shape)
                    accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()
            print("Step number {}".format(steps))
            print("Epoch number {}/{}".format(epoch+1, epochs))
            print("Training Loss is {:.3f}".format(running_loss/print_every))
            print("Validation Loss is {:.3f}".format(validation_loss/len(validation_dataloaders)))
            print("Validation accuracy  is {:.3f}".format(accuracy/len(validation_dataloaders)))
            running_loss=0
            torch.set_default_tensor_type('torch.FloatTensor')
            print("*********")
            model.train()
print("finished==============") 

checkpoint={
    'epochs': epochs,
    'optimizer': optimizer.state_dict,
    'arch': 'vgg16',
   # 'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'class_to_idx': training_datasets.class_to_idx
}

save_dir='trained_model_001.pth'
if(args.save_dir is not None):
    save_dir= args.save_dir

#fullPathToFile = os.path.join(args.save_dir, "trained_model.pth")

torch.save(checkpoint, save_dir)