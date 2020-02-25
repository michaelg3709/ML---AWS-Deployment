import os
import torch
import cv2 
import random

from PIL import Image, ImageFile

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


#Check if gpu support is available
use_cuda = torch.cuda.is_available()


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def predict_breed(image_path, model_tl, class_names, img_data, std_norm):
    '''
    Function that takes a path to an image as input 
    and returns the dog breed that is predicted by the model.    
    Args:
        image_path: path to an image 
        model_tl: transfer learning model
        class_names: class names of dogs
        img_data: transformed dataset images
    Returns:
        Image index, it's class name and image
    '''  
    
    image = Image.open(image_path)

    # Define transformations for the image
    preprocess = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    std_norm
                                     ])
                                
    # Preprocess the image
    img_tensor = preprocess(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    img_tensor = img_tensor.unsqueeze_(0)

    # Input to the network needs to be an autograd Variable
    img_tensor = Variable(img_tensor)
    if use_cuda:
        img_tensor = Variable(img_tensor.cuda())        
    
    model_tl.eval()

    # Our prediction will be the index of the class label with the largest value.
    predict_index = torch.argmax(model_tl(img_tensor)) 
    
    return class_names[predict_index], img_data['train'].classes[predict_index]

