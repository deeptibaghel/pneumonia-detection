# version inspection
import detectron2
print(f"Detectron2 version is {detectron2.__version__}")

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import numpy as np

# create config
cfg = get_cfg()
# below path applies to current installation location of Detectron2
cfgFile = "/usr/local/lib/python3.8/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
cfg.merge_from_file(cfgFile)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "model_final.pth"
cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy

# create predictor
predictor = DefaultPredictor(cfg)

# make prediction
image = cv2.imread('./test_img.jpg')
output = predictor(image)
print(output)