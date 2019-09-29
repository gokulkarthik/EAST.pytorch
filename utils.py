from config import Config
import torch
import numpy as np
import cv2
from sympy import Polygon
from sympy.geometry import intersection
from tqdm import tqdm

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}

score_threshold = config['score_threshold']
nms_method = config['nms_method']
iou_threshold = config['iou_threshold']
max_boxes = config['max_boxes']


def compute_iou_using_sympy(gmap_a, gmap_b):
    
    gmap_a = [(x, y) for x, y in gmap_a.reshape(-1, 2)]
    gmap_b = [(x, y) for x, y in gmap_b.reshape(-1, 2)]
     
    poly_a = Polygon(*gmap_a)
    poly_b = Polygon(*gmap_b)
    
    #print([map(sympy.Float, p) for p in poly_a.vertices])
    #print([map(sumpy.Float, p) for p in poly_b.vertices])

    intersection_ = intersection(poly_a, poly_b)
    #print(intersection_)
    #print([tuple(p) for p in intersection_.vertices])
    area_int = np.abs(np.float(intersection(poly_a, poly_b)))
    area_un = np.abs(np.float(poly_a.area)) + np.abs(np.float(poly_b.area)) + area_int + 10e-6
    iou = area_int/area_un
    
    return iou


def compute_iou_using_cv2(gmap_a, gmap_b):
    
    gmap_a = gmap_a.reshape(-1, 2).astype(np.int32)
    gmap_b = gmap_b.reshape(-1, 2).astype(np.int32)
     
    ref_map_a = np.zeros(shape=(512, 512))
    ref_map_b = np.zeros_like(ref_map_a)
    ref_map_un = np.zeros_like(ref_map_a)

    cv2.fillPoly(ref_map_a, [gmap_a], 1)
    cv2.fillPoly(ref_map_b, [gmap_b], 1)
    cv2.fillPoly(ref_map_un, [gmap_a, gmap_b], 1)

    area_a = ref_map_a.sum()
    area_b = ref_map_b.sum()
    area_un = ref_map_un.sum()
    area_int = area_a + area_b - area_un
    iou = area_int/area_un
    
    #print(area_a, area_b, area_un, area_int, iou)
    
    return iou


def check_overlap(gmap_a, gmap_b):

    gmap_a = [(x, y) for x, y in gmap_a.reshape(-1, 2)]
    gmap_b = [(x, y) for x, y in gmap_b.reshape(-1, 2)]
     
    poly_a = Polygon(*gmap_a)
    poly_b = Polygon(*gmap_b)

    for point in poly_a.vertices:
        if poly_b.encloses(point):
            return 1

    return -1


if nms_method == "iou":
    filter_function = compute_iou_using_cv2
    max_threshold = iou_threshold
elif nms_method == "overlap":
    filter_function = check_overlap
    max_threshold = 0

def non_maximal_supression(score_maps_pred, geometry_maps_pred, score_threshold=0.7, iou_threshold=0.4, max_boxes=10):
    """
    score_maps_pred: [m, 1, 128, 128]
    geometry_maps_pred: [m, 8, 128, 128]
    """
    mini_batch_boxes_pred = []
    for score_map_pred, geometry_map_pred in zip(score_maps_pred, geometry_maps_pred): 
        # score_map_pred: [1, 128, 128]; geometry_map_pred: [8, 128, 128]
        
        score_mask = score_map_pred > score_threshold # [1, 128, 128]
        #print(score_mask)
        score_mask_repeat = np.repeat(score_mask, 8, axis=0) # [8, 128, 128]
        #print(score_mask_repeat)
        
        score_map_pred_selected = score_map_pred[score_mask] # [sel]
        #print()
        #print(score_map_pred_selected.shape)
        #print(score_map_pred_selected)
        selection_order = np.argsort(score_map_pred_selected)[::-1] # [sel]
        #print(selection_order)
        
        geometry_map_pred_selected = geometry_map_pred[score_mask_repeat].reshape(8, -1).T # [-1, 8]
        #print()
        #print(geometry_map_pred_selected.shape)
        #print(geometry_map_pred_selected)
        geometry_map_pred_selected = geometry_map_pred_selected[selection_order] # [-1, 8]
        #print(geometry_map_pred_selected)
        
        if len(geometry_map_pred_selected):
            geometry_map_pred_filtered = [geometry_map_pred_selected[0]]
            for gmap1 in geometry_map_pred_selected[1:]: # hiring

                hired = True
                for gmap2 in geometry_map_pred_filtered: # Existing

                    val = filter_function(gmap1, gmap2)
                    if val >= max_threshold:
                        hired = False
                        break

                if hired == True:
                    geometry_map_pred_filtered.append(gmap1)
                    if len(geometry_map_pred_filtered) >= max_boxes:
                        break

            #geometry_map_pred_filtered = geometry_map_pred_filtered[:max_boxes]
            mini_batch_boxes_pred.append(np.array(geometry_map_pred_filtered).astype(np.int).tolist())
    
    return mini_batch_boxes_pred


def send_message(slack_client, channel, message): 
    
    response = slack_client.chat_postMessage(
        channel=channel,
        text=message,
        username='Deep Updater',
        icon_emoji=':robot_face:')
    
    return response


def send_picture(slack_client, channel, title, picture, message=""): 
    
    response = slack_client.files_upload(
        channels=channel,
        title=title,
        file=picture,
        message=message,
        username='Deep Updater',
        icon_emoji=':robot_face:')
    
    return response


def draw_bbs(image, bbs, color=(0, 0, 255), thickness=1): # BGR
    
    for bb in bbs:
        
        image = cv2.line(image, (bb[0],bb[1]), (bb[2],bb[3]), color, thickness=thickness)
        image = cv2.line(image, (bb[2],bb[3]), (bb[4],bb[5]), color, thickness=thickness)
        image = cv2.line(image, (bb[4],bb[5]), (bb[6],bb[7]), color, thickness=thickness)
        image = cv2.line(image, (bb[6],bb[7]), (bb[0],bb[1]), color, thickness=thickness)
        
    return image


def reverse_shift(geometry_maps_pred): # [8, 128, 128]

	geometry_maps_pred = np.moveaxis(geometry_maps_pred, 0, 2) # [128, 128, 8]
	geometry_maps_pred = geometry_maps_pred.reshape(128, 128, 4, 2)

	for i in range(128):
		for j in range(128):
			geometry_maps_pred[i, j] = geometry_maps_pred[i, j] + np.array([4*i, 4*j])

	geometry_maps_pred = geometry_maps_pred.reshape(128, 128, 8)
	geometry_maps_pred = np.moveaxis(geometry_maps_pred, 2, 0) # [8, 128, 128]

	return geometry_maps_pred


# test code
"""
score_maps_pred = torch.randn([2, 1, 3, 3])
geometry_maps_pred = torch.randn([2, 8, 3, 3])
boxes_pred = non_maximal_supression(score_maps_pred, geometry_maps_pred, score_threshold=0.7, iou_threshold=0.4)
for box_pred in boxes_pred:
    print(np.array(box_pred).shape)
    print(box_pred)
"""