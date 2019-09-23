from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import cv2
import csv
import numpy as np

from tqdm import tqdm
import time

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}
geometry = config['geometry']
label_method = config['label_method']
image_size = config['image_size']
n_H, n_W, n_C = image_size

if geometry == "RBOX":
    raise NotImplementedError("Only implemented for the QUAD geometry")
if label_method == "multiple":
    raise NotImplementedError("Only implemented for the single label method")


def list_images(images_dir):
 
    names = list(os.listdir(images_dir))
    np.random.shuffle(names)
    image_names = names[:config["max_m_train"]]
 
    return image_names


def quads_to_rboxes(quads_coords):

    raise NotImplementedError()


def load_shapes_coords(annotation_path):

    """
    > correct the order of the coords of a quad
    """

    quads_coords = []
    
    with open(annotation_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            text = line[-1]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            quad_cords = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            quads_coords.append(quad_cords)
    quads_coords = np.array(quads_coords, dtype=np.float32) / 512.0
    
    if geometry == "QUAD":
        shapes_coords = quads_coords
    elif geometry == "RBOX":
        shapes_coords =  quads_to_rboxes(coords) 
    
    return shapes_coords


def load_image(image_path):

    image = cv2.imread(image_path)
    image = image[:, :, ::-1] # BGR to RGB
    image = image.astype(np.float32)
    image = np.moveaxis(image, 2, 0) # channel_last to channel_first
    return image


def load_score_and_geometry_map(annotation_path):

    shapes_coords = load_shapes_coords(annotation_path)
    score_map = np.zeros([128, 128, 1])
    geometry_map = np.zeros([128, 128, 8])
    score_map_raw = np.zeros([n_H, n_W, 1])
    geometry_map_raw = np.zeros([n_H, n_W, 8])
    
    if label_method == "single":
        if geometry == "QUAD":
            
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
                                       
        elif geometry == "RBOX":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid geometry: " + geometry)

    elif label_method == "multiple":
        if geometry == "QUAD":
            raise NotImplementedError()
        elif geometry == "RBOX":
            raise NotImplementedError()            
        else:
            raise ValueError("Invalid geometry: " + geometry)
            
    else:
        raise ValueError("Invalid label method: " + label_method)

    assert score_map.shape == (1, 128, 128)
    assert geometry_map.shape == (8, 128, 128)
    
    return score_map, geometry_map

class ImageDataSet(torch.utils.data.Dataset):

    def __init__(self, images_dir, annotations_dir):

        self.images_dir = images_dir
        self.annotations_dir = annotations_dir 
        self.image_names = list_images(images_dir)

    def __getitem__(self, index):

        image_name = self.image_names[index]
        annotation_name = image_name.split(".")[0] + ".csv"
        image_path = os.path.join(self.images_dir, image_name)
        annotation_path = os.path.join(self.annotations_dir, annotation_name)
        # image -> [3, 512, 512]
        image = load_image(image_path)             
        # score_map -> [1, 128, 128]; geometry_map -> [8, 128, 128]
        score_map, geometry_map = load_score_and_geometry_map(annotation_path)
        
        return image, score_map, geometry_map

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
