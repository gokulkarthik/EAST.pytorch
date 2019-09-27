from config import Config
import os
import torch
import math
from tqdm import tqdm
from dataset import ImageTestDataSet
from model import EAST
from utils import non_maximal_supression

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}

geometry = config['geometry']

test_data_dir = config['test_data_dir']

cuda = config['cuda']

trained_model_file = config['trained_model_file']
test_mini_batch_size = config['test_mini_batch_size']

score_threshold = config['score_threshold']
iou_threshold = config['iou_threshold']
max_boxes = config['max_boxes']

test_images_dir = os.path.join(test_data_dir, "images")
test_images_pred_dir = os.path.join(test_data_dir, "images_pred")
test_annotations_pred_dir = os.path.join(test_data_dir, "annotations_pred")

if not os.path.exists(test_images_pred_dir):
    os.mkdir(test_images_pred_dir)
if not os.path.exists(test_annotations_pred_dir):
    os.mkdir(test_annotations_pred_dir)

testset = ImageTestDataSet(test_images_dir)
test_loader = torch.utils.data.DataLoader(testset, batch_size=test_mini_batch_size, shuffle=True)

model = EAST(geometry=geometry)
if cuda:
    model.cuda()
model.load_state_dict(torch.load(trained_model_file))
model.eval()

with torch.no_grad():
    boxes_pred = []
    n_mini_batches = math.ceil(len(testset)/test_mini_batch_size)
    for i, test_egs in tqdm(enumerate(test_loader, start=1), total=n_mini_batches, desc="Testing Mini Batches:"):

        image_names, images = test_egs  
        if cuda:
            images = images.cuda()

        score_maps_pred, geometry_maps_pred = model.forward(images)
        
        score_maps_pred = score_maps_pred.cpu().numpy()
        geometry_maps_pred = geometry_maps_pred.cpu().numpy()

        mini_batch_boxes_pred = non_maximal_supression(score_maps_pred, 
                                                       geometry_maps_pred, 
                                                       score_threshold=score_threshold, 
                                                       iou_threshold=iou_threshold, max_boxes=max_boxes)
        boxes_pred.append(mini_batch_boxes_pred)

    print(boxes_pred)