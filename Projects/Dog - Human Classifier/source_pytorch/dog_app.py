import os
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from source_pytorch.helpers import face_detector, predict_breed
from source_pytorch.dog_detector import dog_detector

def main_detector(img_path, model_tl, class_names, img_data, std_norm):
    '''
    Use pre-trained model to to check if the image at the given path
    contains a human being or a dog or none. 
    
    Args:
        img_path: path to an image
        model_tl: transfer learning model
        class_names: class name of dogs
        img_data: transformed image dataset
        
    Returns:
        print if a human or dog is detected
        print the dog breed or show that neither human face nor a dog detected 
    '''            
    is_human = face_detector(img_path)
    is_dog = dog_detector(img_path)
    breed, name = predict_breed(img_path, model_tl, class_names, img_data, std_norm)
    
    fig = plt.figure(figsize=(16,4))
    
    if(is_human):
        print("Hey... What's up HUMAN?!")
        ax = fig.add_subplot(1,2,1)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        plt.axis('off')

        # display sample of matching breed images
        subdir = '/'.join(['dog_images/valid', str(name)])
        file = random.choice(os.listdir(subdir))
        path = '/'.join([subdir, file])
        ax = fig.add_subplot(1,2,2)
        img = mpimg.imread(path)
        ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')
        plt.title(breed)
        plt.axis('off')
        plt.show()   
        print("You look like ..." + breed)
        print("\n"*3)
        return
    
    elif(is_dog):
        print("Hey... What's up DOG?!")
        ax = fig.add_subplot(1,2,1)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        plt.axis('off')

        # display sample of matching breed images
        subdir = '/'.join(['dog_images/valid', str(name)])
        file = random.choice(os.listdir(subdir))
        path = '/'.join([subdir, file])
        ax = fig.add_subplot(1,2,2)
        img = mpimg.imread(path)
        ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')
        plt.title(breed)
        plt.axis('off')
        plt.show()   
        print("You look like ... " + breed)
        print("\n"*3)
        return
    
    else:
        print('I can\'t determine what you are!')
        ax = fig.add_subplot(1,2,1)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        plt.axis('off')
        plt.show()    
        print("\n"*3)
        return