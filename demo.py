import cv2
import torch
import numpy as np
import onnxruntime
import argparse
from utils_pose import *
from utils_action import *


parser = argparse.ArgumentParser()
parser.add_argument("--input_video", "-iv", type=str, required=True, help="input_video")
parser.add_argument("--weights_pose", "-wp", type=str, default="weights/yolov8l-pose.onnx", help="weights_pose")
parser.add_argument("--weights_action", "-wa", type=str, default="weights/DDNet.onnx", help="weights_action")
opt = parser.parse_args()


class PoseEstimater(object):
    def __init__(self, model) -> None:
        self.providers = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model, providers=self.providers)
    
    def __pre_process(self, image):
        im = letterbox(image, 640, stride=32, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im
    
    def postprocess(self, orig_img):
        """Return detection results for a given input image or list of images."""
        img = self.__pre_process(orig_img)
        preds = self.session.run(None, {self.session.get_inputs()[0].name:np.asarray(img)})
        preds_tens = torch.tensor(preds[0])
        preds = non_max_suppression(preds_tens, 0.25, 0.7)
        for i, pred in enumerate(preds):
            orig_img = orig_img
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            bbox = pred[:, :4].to(torch.int32).tolist()
            pred_kpts = pred[:, 6:].view(len(pred), 17, 3) if len(pred) else pred[:, 6:]
            pred_kpts = scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            xy = pred_kpts[..., :2]
            xy[..., 0] /= orig_img.shape[1]
            xy[..., 1] /= orig_img.shape[0]
            return xy.squeeze().numpy(), bbox[0]


C = Config()
class ActionDetector(object):
    def __init__(self, model) -> None:
        self.providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model, providers=self.providers)

    def detect(self, video):  # video shape (40, 17, 2)
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
    model_pose = PoseEstimater(model=opt.weights_pose)
    model_action = ActionDetector(model=opt.weights_action)
    cap = cv2.VideoCapture(opt.input_video)

    frame_seq = []
    output_action_idx = 1
    target_dict = ["climb", "no climb"]

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            output_pose, bbox = model_pose.postprocess(frame)
            if output_pose.shape == (17, 2):
                frame_seq.append(output_pose)
            if len(frame_seq) == 40:
                start_time = time.time()
                output_action = model_action.detect(np.array(frame_seq))
                duration = time.time() - start_time
                output_action_idx = np.argmax(output_action)
                print(f"Action: {output_action_idx} - Time: {duration:.4f}s")
                frame_seq = frame_seq[1:]
            
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
            frame = cv2.putText(frame, 
                                target_dict[output_action_idx], 
                                (bbox[0], bbox[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(25) & 0xff == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
