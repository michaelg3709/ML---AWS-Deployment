import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from PIL import Image

# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    vgg16 = vgg16.cuda()
    
# Function to predict classes using VGG16 model
def vgg16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path    
    Args:
        img_path: path to an image        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''   
    
    # Load the image from provided path
    img = Image.open(img_path) 
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize])
    img_tensor = preprocess(img).float()
    
    # PyTorch pretrained models expect the Tensor dims to be (num input imgs, num color channels, height, width).
    # Currently however, we have (num color channels, height, width); let's fix this by inserting a new axis.
    # Insert the new axis at index 0 i.e. in front of the other axes/dims.
    img_tensor.unsqueeze_(0)  
    
    # Now that we have preprocessed our img, we need to convert it into a variable
    # A PyTorch Variable is a wrapper around a PyTorch Tensor.
    # The input to the network needs to be an autograd Variable  
    img_tensor = Variable(img_tensor)  
    if use_cuda:
        img_tensor = Variable(img_tensor.cuda())
        
    vgg16.eval()

    # Prediction will be the index of the class label with the largest value.
    predict_index = torch.argmax(vgg16(img_tensor)) 
    return predict_index # predicted class index


def dog_detector(img_path):
    prediction = vgg16_predict(img_path)
    return ((prediction <= 268)) & (prediction >= 151)