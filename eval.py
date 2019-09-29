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
from utils import non_maximal_supression, draw_bbs, reverse_shift
import math
import cv2

do_eval_trainset = False
do_eval_devset = True

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}

geometry = config['geometry']
label_method = config['label_method']
use_formatted_data = config['use_formatted_data']

train_data_dir = config['train_data_dir']
dev_data_dir = config['dev_data_dir']

cuda = config['cuda']
smoothed_l1_loss_beta = config["smoothed_l1_loss_beta"]

trained_model_file = config['trained_model_file']
eval_mini_batch_size = config['eval_mini_batch_size']

score_threshold = config['score_threshold']
iou_threshold = config['iou_threshold']
max_boxes = config['max_boxes']

representation = geometry + "_" + label_method

model = EAST(geometry=geometry)
loss_function = LossFunction()
if cuda:
    model.cuda()
    loss_function.cuda()
model.load_state_dict(torch.load(trained_model_file))
model.eval()

def eval_dataset(data_dir):
    
    data_images_dir = os.path.join(data_dir, "images")
    data_annotations_dir = os.path.join(data_dir, "annotations")
    if use_formatted_data:
        data_annotations_formatted_dir = data_annotations_dir + "_" + representation
    data_images_pred_dir = os.path.join(data_dir, "images_pred")
    data_annotations_pred_dir = os.path.join(data_dir, "annotations_pred")
    
    if not os.path.exists(data_images_pred_dir):
        os.mkdir(data_images_pred_dir)
    if not os.path.exists(data_annotations_pred_dir):
        os.mkdir(data_annotations_pred_dir)
    
    dataset = ImageDataSet(data_images_dir, data_annotations_formatted_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=eval_mini_batch_size, shuffle=True)

    score_loss, geometry_loss, loss = 0, 0, 0
    boxes_pred = []
    n_mini_batches = math.ceil(len(dataset)/eval_mini_batch_size)
    for i, data_egs in tqdm(enumerate(data_loader, start=1), total=n_mini_batches, desc="Evaluating Mini Batches:"):
        
        image_names, images, score_maps, geometry_maps = data_egs
        if cuda:
            images = images.cuda()
            score_maps = score_maps.cuda()
            geometry_maps = geometry_maps.cuda()
            
        score_maps_pred, geometry_maps_pred = model.forward(images)
  
        mini_batch_loss = loss_function.compute_loss(score_maps.double(), 
            score_maps_pred.double(),
            geometry_maps.double(), 
            geometry_maps_pred.double(),
            smoothed_l1_loss_beta = smoothed_l1_loss_beta)

        mini_batch_loss_of_score_item = loss_function.loss_of_score.item()
        mini_batch_loss_of_geometry_item = loss_function.loss_of_geometry.item()
        mini_batch_loss_item = mini_batch_loss.item()
              
        score_loss += mini_batch_loss_of_score_item
        geometry_loss += mini_batch_loss_of_geometry_item
        loss += mini_batch_loss_item
        
        score_maps_pred = score_maps_pred.cpu().numpy()
        geometry_maps_pred = geometry_maps_pred.cpu().numpy()

        if representation == "QUAD_multiple":
        	geometry_maps_pred = reverse_shift(geometry_maps_pred) # [8, 128, 128]

        #print("NMS Started")
        nms_tic = time.time()
        
        mini_batch_boxes_pred = non_maximal_supression(score_maps_pred, 
                                                       geometry_maps_pred, 
                                                       score_threshold=score_threshold, 
                                                       iou_threshold=iou_threshold, 
                                                       max_boxes=max_boxes)

        
        nms_toc = time.time()
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(nms_toc - nms_tic))
        #print("NMS Ended", "Duration", toc-tic)

        boxes_pred.extend(mini_batch_boxes_pred)
        
        for image_name, eg_boxes_pred in zip(image_names, mini_batch_boxes_pred):

            annotation_name = image_name.split(".")[0] + ".csv"
            image_path = os.path.join(data_images_dir, image_name)
            annotation_path = os.path.join(data_annotations_dir, annotation_name)
            image_pred_path = os.path.join(data_images_pred_dir, image_name)
            annotation_pred_path = os.path.join(data_annotations_pred_dir, annotation_name)
            
            image = cv2.imread(image_path)
            geometry_map = pd.read_csv(annotation_path, header=None).iloc[:,:-1].values.tolist()
            
            image = draw_bbs(image, geometry_map, color=(255, 0, 0)) #BGR
            image = draw_bbs(image, eg_boxes_pred, color=(0, 0, 255)) #BGR
            
            cv2.imwrite(image_pred_path, image)             
            eg_boxes_pred = pd.DataFrame(eg_boxes_pred).to_csv(annotation_pred_path, header=False, index=False)
        
    score_loss /= n_mini_batches
    geometry_loss /= n_mini_batches
    loss /= n_mini_batches
    
    print(data_dir)
    message = "Score Loss: {:.6f}; Geo Loss: {:.6f}; Loss: {:.6f}".format(score_loss,
                                                                          geometry_loss,
                                                                          loss)
    print(message)
        
    return boxes_pred
   
    
with torch.no_grad():
    if do_eval_trainset: 
        boxes_pred_trainset = eval_dataset(train_data_dir)
    if do_eval_devset:
        boxes_pred_devset = eval_dataset(dev_data_dir)