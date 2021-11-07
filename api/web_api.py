import flask
from flask_cors import CORS
from flask import request, jsonify
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
import requests
import numpy as np
import base64

def prepare_pridctor():
    # create config
    cfg = get_cfg()
    cfgFile = "/usr/local/lib/python3.8/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
   
    cfg.merge_from_file(cfgFile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "./model_final.pth"
    cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized.")
    classes = ["pneumonia"]
    return (predictor, classes)

app = flask.Flask(__name__)
CORS(app)
predictor, classes = prepare_pridctor()

@app.route("/api/predict", methods=["POST"])
def process_score_image_request():
    f = request.files['file']  
    response = f.read()
   
    image = cv2.imdecode(np.fromstring(response, np.uint8), cv2.IMREAD_COLOR)

    # make prediction
    scoring_result = predictor(image)
    instances = scoring_result["instances"]
    scores = instances.get_fields()["scores"].tolist()
    pred_classes = instances.get_fields()["pred_classes"].tolist()
    pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()

    # image download
    v = Visualizer(image[:, :, ::-1],
                  metadata=None, 
                  scale=0.5, 
                #  instance_mode=ColorMode.IMAGE_BW   
    )
    out = v.draw_instance_predictions(instances.to("cpu")) 
    img_string = base64.b64encode(cv2.imencode('.jpg', out.get_image()[:, :, ::-1])[1]).decode()

    response = {
        "scores": scores,
        "pred_classes": pred_classes,
        "pred_boxes" : pred_boxes,
        "classes": classes,
        "image": img_string
    }
    return jsonify(response)

app.run(host="0.0.0.0", port=5000)