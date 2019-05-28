from albumentations import Compose, Normalize, CenterCrop
from albumentations.torch import ToTensor

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_transform_base(size):
    return Compose([CenterCrop(size, size),
                    Normalize(mean, std)], 
                   to_tensor=ToTensor())

def get_transform_kp(size): 
    return Compose([CenterCrop(size, size), 
                    Normalize(mean, std)],
                   to_tensor=ToTensor(),
                   keypoint_params={'format': 'xy', 'remove_invisible': False})
                   
#                     additional_targets={'image1': 'image',
#                                         'image2':'image',
#                                         'keypoints1': 'keypoints',
#                                        'keypoints':'keypoints'})