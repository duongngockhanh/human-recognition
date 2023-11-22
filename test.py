import torch
import numpy as np
import onnxruntime
import cv2
from utils import *

class Config:
    def __init__(self):
        self.frame_l = 40
        self.joint_n = 18
        self.joint_d = 2
        self.clc_num = 2
        self.feat_d = 153
        self.filters = 16

C = Config()

class Detector(object):
    def __init__(self, model="weights/model.onnx") -> None:
        self.providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model, providers=self.providers)

    def detect(self, video):  # video shape (40, 18, 2)
        p = zoom(video, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)
        M = get_CG(p, C)
        p = np.asarray(p, dtype=np.float32)
        p = np.expand_dims(p, axis=0)
        M = np.asarray(M, dtype=np.float32)
        M = np.expand_dims(M, axis=0)
        output = self.session.run(
            None,
            {
                self.session.get_inputs()[0].name: M,
                self.session.get_inputs()[1].name: p,
            }
        )
        return output

if __name__ == "__main__":
    video = np.random.rand(40, 18, 2)
    model = Detector("model.onnx")
    output = model.detect(video)
    print("Action:", np.argmax(output))