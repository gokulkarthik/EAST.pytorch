from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import cv2
import csv

"""
Reference: https://github.com/liushuchun/EAST.pytorch
"""


config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}
geometry = config['geometry']
if geometry == "RBOX":
	raise NotImplementedError("Only implemented for the QUAD geometry")


def list_images(images_dir):
 
    names = list(os.listdir(images_dir))
    np.random.shuffle(names)
    image_names = names[:config["max_m_train"]]
 
    return image_names


def quads_to_rboxes(quads_coords):

	pass


def load_coords(annotation_path):

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
            quads_coords_c.append(quad_cords)
    quads_coords = np.array(quads_coords, dtype=np.float32)
    
    if geometry == "QUAD":
    	coords = quads_coords
    elif geometry == "RBOX":
    	coords =  quads_to_rboxes(coords) 
    
    return coords

def load_image(image_path):

	image = cv2.imread(image_path)
	image = image[:, :, ::-1] # BGR to RGB

	return image


def load_score_and_geometry_map(annotation_path):

	coords = load_coords(annotation_path)

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
    	image = load_image(image_path)
    	score_map, geometry_map = load_score_and_geometry_map(annotation_path)
        
        return image, score_map, geometry_map

    def __len__(self):
        
        return len(self.image_names)