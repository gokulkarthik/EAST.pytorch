# EAST Text Detection Model in PyTorch
### - with single QUAD Representation of Bounding Boxes 
A PyTorch implementation of EAST: An Efficient and Accurate Scene Text Detector for bounding box detection

## References: 

    1. https://arxiv.org/pdf/1704.03155.pdf
    2. https://github.com/liushuchun/EAST.pytorch
    3. https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

## Steps:

    1. Data folder should be organised as follows:
    The data dir should have 3 sub dirs namely, "train", "dev" and "test".
    The "train" and "dev" dirs should have 2 sub dirs namely "images" and "annotations"
    The "test" dir should have a sub dir "images"
    
    2. Run format_data.py
    
    3. Train the model using train.py
    
    4. Evaluate the model using eval.py
    
    5. Predict using test.py
