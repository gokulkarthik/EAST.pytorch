from config import Config
import os
import torch
import math
from tqdm import tqdm
from dataset import ImageTestDataSet
from model import EAST
from utils import non_maximal_supression, reverse_shift

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}

geometry = config['geometry']
label_method = config['label_method']

test_data_dir = config['test_data_dir']

cuda = config['cuda']

trained_model_file = config['trained_model_file']
test_mini_batch_size = config['test_mini_batch_size']

score_threshold = config['score_threshold']
iou_threshold = config['iou_threshold']
max_boxes = config['max_boxes']

representation = geometry + "_" + label_method

test_images_dir = os.path.join(test_data_dir, "images")
test_images_pred_dir = os.path.join(test_data_dir, "images_pred")
test_annotations_pred_dir = os.path.join(test_data_dir, "annotations_pred")

model = EAST(geometry=geometry)
if cuda:
    model.cuda()
model.load_state_dict(torch.load(trained_model_file))
model.eval()

if not os.path.exists(test_images_pred_dir):
    os.mkdir(test_images_pred_dir)
if not os.path.exists(test_annotations_pred_dir):
    os.mkdir(test_annotations_pred_dir)

testset = ImageTestDataSet(test_images_dir)
test_loader = torch.utils.data.DataLoader(testset, batch_size=test_mini_batch_size, shuffle=True)

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

        if representation == "QUAD_multiple":
        	geometry_maps_pred = reverse_shift(geometry_maps_pred) # [8, 128, 128]

        mini_batch_boxes_pred = non_maximal_supression(score_maps_pred, 
                                                       geometry_maps_pred, 
                                                       score_threshold=score_threshold, 
                                                       iou_threshold=iou_threshold, max_boxes=max_boxes)
        boxes_pred.extend(mini_batch_boxes_pred)

        for image_name, eg_boxes_pred in zip(image_names, mini_batch_boxes_pred):

            annotation_name = image_name.split(".")[0] + ".csv"
            image_path = os.path.join(data_images_dir, image_name)
            #annotation_path = os.path.join(data_annotations_dir, annotation_name)
            image_pred_path = os.path.join(data_images_pred_dir, image_name)
            annotation_pred_path = os.path.join(data_annotations_pred_dir, annotation_name)
            
            image = cv2.imread(image_path)
            #geometry_map = pd.read_csv(annotation_path, header=None).iloc[:,:-1].values.tolist()
            
            #image = draw_bbs(image, geometry_map, color=(255, 0, 0)) #BGR
            image = draw_bbs(image, eg_boxes_pred, color=(0, 0, 255)) #BGR
            
            cv2.imwrite(image_pred_path, image)             
            eg_boxes_pred = pd.DataFrame(eg_boxes_pred).to_csv(annotation_pred_path, header=False, index=False)

    print(boxes_pred)