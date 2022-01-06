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
import predict_functions

parser = argparse.ArgumentParser(description='Predict the class of an image')
parser.add_argument('--topk', default=5, type=int)
parser.add_argument('--category_names', default='cat_to_name.json')
parser.add_argument('--gpu', default=False, type=bool)
parser.add_argument('path_to_img', default='flowers/test/100/image_07896.jpg', type=str)
parser.add_argument('path_to_chekpoint', default='trained_model_002.pth', type=str)


print("Loading the checkpoint")
model=predict_functions.load_checkpoint(args.path_to_chekpoint)



#try to predict
print("Predicting")
probs, labels, flowers =predict_functions.predict(args.path_to_img, model, args.topk, args.gpu) 

plt.figure(figsize = (6,10))
ax = plt.subplot(2,1,1)
plt.subplot(2,1,2)
sb.barplot(x=probs, y=flowers, color=sb.color_palette()[0]);
plt.show()
