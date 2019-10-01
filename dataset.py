from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import cv2
import csv
import numpy as np
import pandas as pd

from tqdm import tqdm
import time

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}

image_size = config['image_size']
geometry = config['geometry']
label_method = config['label_method']
use_formatted_data = config['use_formatted_data']

max_m_train = config['max_m_train']

representation = geometry + "_"+ label_method

n_H, n_W, n_C = image_size

if geometry == "RBOX":
    raise NotImplementedError("Only implemented for the QUAD geometry")
if label_method == "multiple":
    raise NotImplementedError("Only implemented for the single label method")


def list_images(images_dir, store=False):
 
    names = list(os.listdir(images_dir))
    image_names = names[:max_m_train]
    np.random.shuffle(image_names)    

    if store:
        data_dir = "/".join(images_dir.split("/")[:-1])
        file = os.path.join(data_dir, 'train_image_names.csv')
        with open(file, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(image_names)
 
    return image_names


def quads_to_rboxes(quads_coords):

    raise NotImplementedError()


def load_shapes_coords(annotation_path):

    """
    > correct the order of the coords of a quad
    """

    quads_coords = pd.read_csv(annotation_path, header=None)
    quads_coords = quads_coords.iloc[:,:-1].values # [n_box, 8]
    quads_coords = quads_coords.reshape(-1, 4, 2)
    
    if geometry == "QUAD":
        shapes_coords = quads_coords
    elif geometry == "RBOX":
        shapes_coords =  quads_to_rboxes(coords)
    else:
        raise ValueError("Invalid Geometry")
    
    return shapes_coords


def load_image(image_path):

    image = cv2.imread(image_path)
    image = image[:, :, ::-1] # BGR to RGB
    image = image.astype(np.float32)
    image = np.moveaxis(image, 2, 0) # channel_last to channel_first
    return image


def load_score_and_geometry_map_raw(annotation_path):

    shapes_coords = load_shapes_coords(annotation_path)
    score_map = np.zeros([1, 128, 128])
    geometry_map = np.zeros([8, 128, 128])
    score_map_raw = np.zeros([n_H, n_W, 1])
    geometry_map_raw = np.zeros([n_H, n_W, 8])
    
    if representation == "QUAD_single":
            
        shapes_centre = np.mean(shapes_coords, axis=1).astype(np.int32)
        #print("shapes_coords", shapes_coords.shape, "\n", shapes_coords)
        #print("shapes_centre", shapes_centre.shape, "\n", shapes_centre)
        for shape_coords, shape_centre in zip(shapes_coords, shapes_centre): # shape_coords -> [4, 2], shape_centre -> [2]
            c_h, c_w = shape_centre
            score_map_raw[c_h, c_w, 0] = 1
            geometry_map_raw[c_h, c_w] = shape_coords.flatten() # [8]

        score_map_raw = np.moveaxis(score_map_raw, 2, 0) # channel_last to channel_first
        geometry_map_raw = np.moveaxis(geometry_map_raw, 2, 0) # channel_last to channel_first

        max_pool_2d = nn.MaxPool2d((4,4), stride=4)
        score_map_raw = torch.from_numpy(score_map_raw)
        score_map = max_pool_2d(score_map_raw)
        geometry_map_raw = torch.from_numpy(geometry_map_raw)
        geometry_map = max_pool_2d(geometry_map_raw)

        #print("score_map", score_map.shape, "\n", score_map.sum())
        #print("geometry_map", geometry_map.shape, "\n", (geometry_map.sum(axis=0)>0).sum())
        #time.sleep(10)
                                       
    elif representation == "QUAD_multiple":
        
        raise NotImplementedError()
        
    elif representation == "RBOX_single":
        
        raise NotImplementedError()
        
    elif representation == "RBOX_multiple":
        
        raise NotImplementedError()
            
    else:
        
        raise ValueError("Invalid representation: " + representation)

    assert score_map.shape == (1, 128, 128)
    assert geometry_map.shape == (8, 128, 128)
    
    return score_map, geometry_map


def load_score_and_geometry_map_formatted(annotation_path):
    
    score_map = np.zeros([1, 128, 128])
    geometry_map = np.zeros([8, 128, 128])
    
    if representation == "QUAD_single" or representation == "QUAD_multiple":
        
        geometry_map = pd.read_csv(annotation_path, header=None).values # [(128*128), 8]
        geometry_map = geometry_map.reshape(128, 128, 8)
        geometry_map = np.moveaxis(geometry_map, 2, 0)
        score_map = (geometry_map.sum(axis=0) > 0).astype(np.int).reshape(1, 128, 128)
        
    elif representation == "RBOX_single":
        
        raise NotImplementedError()
        
    elif representation == "RBOX_multiple":
        
        raise NotImplementedError()
            
    else:
        
        raise ValueError("Invalid representation: " + representation)
        
    assert score_map.shape == (1, 128, 128)
    assert geometry_map.shape == (8, 128, 128)
    
    return score_map, geometry_map
    

class ImageDataSet(torch.utils.data.Dataset):

    def __init__(self, images_dir, annotations_dir):

        self.images_dir = images_dir
        self.annotations_dir = annotations_dir 
        self.image_names = list_images(images_dir, store=True)

    def __getitem__(self, index):

        image_name = self.image_names[index]
        image_path = os.path.join(self.images_dir, image_name)
        image = load_image(image_path) # image -> [3, 512, 512]
        
        annotation_name = image_name.split(".")[0] + ".csv"
        annotation_path = os.path.join(self.annotations_dir, annotation_name)
        
        if use_formatted_data:
            score_map, geometry_map = load_score_and_geometry_map_formatted(annotation_path) # score_map -> [1, 128, 128]; geometry_map -> [8, 128, 128]
        else:
            score_map, geometry_map = load_score_and_geometry_map_raw(annotation_path) # score_map -> [1, 128, 128]; geometry_map -> [8, 128, 128]
        
        return image_name, image, score_map, geometry_map

    def __len__(self):
        
        return len(self.image_names)
    
    
class ImageTestDataSet(torch.utils.data.Dataset):

    def __init__(self, images_dir):

        self.images_dir = images_dir
        self.image_names = list_images(images_dir, store=False)

    def __getitem__(self, index):

        image_name = self.image_names[index]
        image_path = os.path.join(self.images_dir, image_name)
        # image -> [3, 512, 512]
        image = load_image(image_path)             
        
        return image_name, image

    def __len__(self):
        
        return len(self.image_names)


# test code
"""
train_data_dir = config["train_data_dir"]
mini_batch_size = config["mini_batch_size"]
train_images_dir = os.path.join(train_data_dir, "images")
train_annotations_dir = os.path.join(train_data_dir, "annotations")

trainset = ImageDataSet(train_images_dir, train_annotations_dir)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True)
print("Number of examples:", len(trainset))
print("Mini batch size:", mini_batch_size)
n_mini_batches = len(trainset)//mini_batch_size + int(len(trainset)%mini_batch_size!=0)
print("Number of mini batches:", n_mini_batches) 

for i, train_eg in tqdm(enumerate(train_loader), total=n_mini_batches):
    image, score_map, geometry_map = train_eg      
    print(image.size())
    print(score_map.size(), geometry_map.size())
    time.sleep(10)
"""
