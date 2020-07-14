from .face_detect_mtcnn import FaceDetector as FaceDetector_mtcnn
from .face_detect_ssd import FaceDetector as FaceDetector_ssd
from .face_detect_onnx import Detector as FaceDetector_onnx

__all_ = ['FaceDetector_mtcnn', 'FaceDetector_ssd', 'FaceDetector_onnx']
