from config import Config
import torch
import numpy as np

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}

score_threshold = config['score_threshold']
iou_threshold = config['iou_threshold']
max_boxes = config['max_boxes']

def compute_iou(gmap_a, gmap_b):
    """
    warning: This is just an approximation to IoU
    """
    x1_a, y1_a, x3_a, y3_a = gmap_a[0], gmap_a[1], gmap_a[4], gmap_a[5]
    x1_b, y1_b, x3_b, y3_b = gmap_b[0], gmap_b[1], gmap_b[4], gmap_b[5]
    x1_int, y1_int, x3_int, y3_int = max(x1_a, x1_b), max(y1_a, y1_b), min(x3_a, x3_b), min(y3_a, y3_b)
    
    area_a = np.abs((x3_a-x1_a) * (y3_a-y1_a))
    area_b = np.abs((x3_b-x1_b) * (y3_b-y1_b))
    area_int = np.abs((x3_int-x1_int) * (y3_int-y1_int))
    area_un = np.abs(area_a + area_b - area_int) + 10e-6
    iou = area_int/area_un
    
    return iou

def non_maximal_supression(score_maps_pred, geometry_maps_pred, score_threshold=0.7, iou_threshold=0.4, max_boxes=10):
    """
    score_maps_pred: [m, 1, 128, 128]
    geometry_maps_pred: [m, 8, 128, 128]
    """
    score_maps_pred = score_maps_pred.cpu().numpy()
    geometry_maps_pred = geometry_maps_pred.cpu().numpy()
    mini_batch_boxes_pred = []
    for score_map_pred, geometry_map_pred in zip(score_maps_pred, geometry_maps_pred): # [1, 128, 128], [8, 128, 128]
        
        score_mask = score_map_pred > score_threshold # [1, 128, 128]
        #print(score_mask)
        score_mask_repeat = np.repeat(score_mask, 8, axis=0) # [8, 128, 128]
        #print(score_mask_repeat)
        
        score_map_pred_selected = score_map_pred[score_mask] # [-1]
        #print(score_map_pred_selected)
        selection_order = np.argsort(score_map_pred_selected)[::-1] # [-1]
        #print(selection_order)
        
        geometry_map_pred_selected = geometry_map_pred[score_mask_repeat].reshape(8, -1).T # [-1, 8]
        #print(geometry_map_pred_selected)
        geometry_map_pred_selected = geometry_map_pred_selected[selection_order] # [-1, 8]
        #print(geometry_map_pred_selected)
        
        if len(geometry_map_pred_selected):
            geometry_map_pred_filtered = [geometry_map_pred_selected[0]]
            for gmap1 in geometry_map_pred_selected[1:]: # hiring

                hired = True
                for gmap2 in geometry_map_pred_filtered: # Existing

                    iou = compute_iou(gmap1, gmap2)
                    if iou >= iou_threshold:
                        hired = False
                        break

                if hired == True:
                    geometry_map_pred_filtered.append(gmap1)

            geometry_map_pred_filtered = geometry_map_pred_filtered[:max_boxes]
            mini_batch_boxes_pred.append(geometry_map_pred_filtered.astype(np.int).tolist())
    
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

# test code
"""
score_maps_pred = torch.randn([2, 1, 3, 3])
geometry_maps_pred = torch.randn([2, 8, 3, 3])
boxes_pred = non_maximal_supression(score_maps_pred, geometry_maps_pred, score_threshold=0.7, iou_threshold=0.4)
for box_pred in boxes_pred:
    print(np.array(box_pred).shape)
    print(box_pred)
"""