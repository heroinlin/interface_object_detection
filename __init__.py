from .face_detect_mtcnn import FaceDetector as FaceDetector_mtcnn, show_bboxes
from .face_detect_ssd import FaceDetector as FaceDetector_ssd, draw_detection_rects, box_transform


FaceDetector = FaceDetector_ssd
