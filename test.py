from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import ImageDataSet
from tqdm import tqdm
import time
from model import EAST
from loss import LossFunction
from utils import non_maximal_supression

do_validate_trainset = False
do_validate_devset = True

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}

geometry = config['geometry']

train_data_dir = config['data_train_dir']
dev_data_dir = config['data_dev_dir']
test_data_dir = config['data_test_dir']

cuda = config['cuda']

test_model_file = config['test_model_file']

test_mini_batch_size = config['test_mini_batch_size']

score_threshold = config['score_threshold']
iou_threshold = config['iou_threshold']
max_boxes = config['max_boxes']

train_images_dir = os.path.join(train_data_dir, "images")
train_annotations_dir = os.path.join(train_data_dir, "annotations")

dev_images_dir = os.path.join(dev_data_dir, "images")
dev_annotations_dir = os.path.join(dev_data_dir, "annotations")

test_images_dir = os.path.join(test_data_dir, "images")
test_annotations_dir = os.path.join(test_data_dir, "annotations")

trainset = ImageDataSet(train_images_dir, train_annotations_dir)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True)

devset = ImageDataSet(dev_images_dir, dev_annotations_dir)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=mini_batch_size, shuffle=True)

testset = ImageTestDataSet(test_images_dir, test_annotations_dir)
test_loader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=True)

model = EAST(geometry=geometry)
if cuda:
    model.cuda()
model.load_state_dict(torch.load(test_model_file))
model.eval()


train_loss = 0
train_score_loss = 0
train_geometry_loss = 0
def validate_trainset():
    n_mini_batches = math.ceil(len(trainset)/test_mini_batch_size)
    boxes_pred = []
    for i, train_egs in tqdm(enumerate(dev_loader, start=1), total=n_mini_batches, desc="Validating Mini Batches of Trainset:"):
        
        images, score_maps, geometry_maps = dev_egs
        if cuda:
            images = Variable(images.cuda())
            score_maps = Variable(score_maps.cuda())
            geometry_maps = Variable(geometry_maps.cuda())
            
        score_maps_pred, geometry_maps_pred = model.forward(images)
  
        mini_batch_loss = loss_function.compute_loss(score_maps.double(), 
            score_maps_pred.double(),
            geometry_maps.double(), 
            geometry_maps_pred.double(),
            smoothed_l1_loss_beta = smoothed_l1_loss_beta)

        mini_batch_loss_of_score_item = loss_function.loss_of_score.item()
        mini_batch_loss_of_geometry_item = loss_function.loss_of_geometry.item()
        mini_batch_loss_item = mini_batch_loss.item()
              
        train_score_loss += mini_batch_loss_of_score_item
        train_geometry_loss += mini_batch_loss_of_geometry_item
        train_loss += mini_batch_loss_item
        
        mini_batch_boxes_pred = non_maximal_supression(score_maps_pred, geometry_maps_pred, score_threshold=score_threshold, iou_threshold=iou_threshold, max_boxes=max_boxes)
        boxes_pred.extend(mini_batch_boxes_pred)
    
    return boxes_pred

    
dev_loss = 0
dev_score_loss = 0
dev_geometry_loss = 0
def validate_devset():
    n_mini_batches = math.ceil(len(devset)/test_mini_batch_size)
    boxes_pred = []
    for i, dev_egs in tqdm(enumerate(dev_loader, start=1), total=n_mini_batches, desc="Validating Mini Batches of Devset:"):
        
        images, score_maps, geometry_maps = dev_egs
        if cuda:
            images = Variable(images.cuda())
            score_maps = Variable(score_maps.cuda())
            geometry_maps = Variable(geometry_maps.cuda())
            
        score_maps_pred, geometry_maps_pred = model.forward(images)
  
        mini_batch_loss = loss_function.compute_loss(score_maps.double(), 
            score_maps_pred.double(),
            geometry_maps.double(), 
            geometry_maps_pred.double(),
            smoothed_l1_loss_beta = smoothed_l1_loss_beta)

        mini_batch_loss_of_score_item = loss_function.loss_of_score.item()
        mini_batch_loss_of_geometry_item = loss_function.loss_of_geometry.item()
        mini_batch_loss_item = mini_batch_loss.item()
              
        dev_score_loss += mini_batch_loss_of_score_item
        dev_geometry_loss += mini_batch_loss_of_geometry_item
        dev_loss += mini_batch_loss_item
        
        mini_batch_boxes_pred = non_maximal_supression(score_maps_pred, geometry_maps_pred, score_threshold=score_threshold, iou_threshold=iou_threshold, max_boxes=max_boxes)
        boxes_pred.append(mini_batch_boxes_pred)
    
    return boxes_pred
   

def test():
    n_mini_batches = math.ceil(len(testset)/test_mini_batch_size)
    boxes_pred
    for i, test_egs in tqdm(enumerate(test_loader, start=1), total=n_mini_batches, desc="Testing Mini Batches:"):

        images = test_egs  
        if cuda:
            images = Variable(images.cuda())

        score_maps_pred, geometry_maps_pred = model.forward(images)
        
        mini_batch_boxes_pred = non_maximal_supression(score_maps_pred, geometry_maps_pred, score_threshold=score_threshold, iou_threshold=iou_threshold, max_boxes=max_boxes)
        boxes_pred.append(mini_batch_boxes_pred)
        
    return boxes_pred


if do_validate_trainset: 
    boxes_pred_trainset = validate_trainset()
if do_validate_testset:
    boxes_pred_devset = validate_devset()
boxes_pred_testset = test()