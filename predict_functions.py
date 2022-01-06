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


def load_checkpoint(filepath):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(filepath, map_location=map_location)
    model=  models.vgg16(pretrained=True)
    #model.arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False
        
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image=Image.open(image)
    #pil_image= pil_image/255
    img_transforms= transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    transformed_img= img_transforms(pil_image)
   
    return  transformed_img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

def predict(image_path, model, topk=5. use_gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img_tensor=(process_image(image_path))
    img_tensor.unsqueeze_(0)
    if torch.cuda.is_available() and use_gpu:
        img_tensor= img_tensor.cuda()
        model= model.cuda()
   
    
    print("#################")
    print((img_tensor.shape))
    model.eval()
    with torch.no_grad():
      
        output= model.forward(img_tensor)
        ps= torch.exp(output).data
        if(use_gpu):
            ps =ps.cpu()
        ps_topk=ps.topk(topk,dim=1)[0].numpy()[0]
        ps_topk_idx= ps.topk(topk)[1].numpy()[0]
        print("^^^^^^^^^^^^^^^^^ " )
        print(ps_topk)
        print("^^^^^^^^^^^^^^^^^ ")
        print(ps_topk_idx)
       
        idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
        top_labels = [idx_to_class[idx] for idx in ps_topk_idx]
        top_flowers = [cat_to_name[lab] for lab in top_labels]

        return ps_topk, top_labels, top_flowers