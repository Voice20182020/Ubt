import onnxruntime
import cv2
import numpy as np

input_img = cv2.imread('demo.jpg').astype(np.float32)
# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

ort_session = onnxruntime.InferenceSession("end2end.onnx")
ort_inputs = {'input': input_img}
ort_output = ort_session.run(['dets','labels'], ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 0]).astype(np.uint8)
cv2.imwrite("end2end.png", ort_output)