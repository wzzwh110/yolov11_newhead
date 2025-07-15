import random

import numpy as np
import torch
from ultralytics import YOLO


def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


if __name__ == '__main__':


  model = YOLO('yolov11_new.yaml')
  set_seed(42)  # 固定随机因子
  # Train the model
  results = model.train(
    data='tct_4.yaml',
    epochs=100,
    batch=128,
    imgsz=768,
    scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
    mosaic=1.0,
    mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
    device="0",
    name = "yolov11new_tct",
  )

  # # Evaluate model performance on the validation set
  # metrics = model.val()
  #
  # # Perform object detection on an image
  # results = model("path/to/image.jpg")
  # results[0].show()
