from .face_detect_mtcnn import FaceDetector as FaceDetector_mtcnn, show_bboxes
from .face_detect_ssd import FaceDetector as FaceDetector_ssd, box_transform
from .face_detect_onnx import Detector as FaceDetector_onnx, draw_detection_rects



FaceDetector = FaceDetector_ssd
