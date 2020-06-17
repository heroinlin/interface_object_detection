import os
from face_detect_ssd import FaceDetector, box_transform, draw_detection_rects
import cv2
working_root = os.path.split(os.path.realpath(__file__))[0]
os.chdir(working_root)


def image_test():
    detector = FaceDetector()
    image_path = os.path.join(os.getcwd(), "../samples/emma_input.jpg")
    image = cv2.imread(image_path, 1)
    bounding_boxes = detector.detect(image)
    image = cv2.resize(image, dsize=(640, 576))
    image = draw_detection_rects(image, bounding_boxes)
    cv2.imshow("image", image)
    cv2.waitKey()
    print(bounding_boxes)


def face_detect_test(video_path):
    face_detector = FaceDetector()

    cap = cv2.VideoCapture()
    if not cap.open(video_path):
        print("video_path is not valid!")
        return -1
    while True:
        ret, img = cap.read()
        if not ret:
            break
        bounding_boxes = face_detector.detect(img)
        image = draw_detection_rects(img.copy(), bounding_boxes)

        # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow('0', image)
        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    video_path = r"F:\Database\face_videos\0188_03_021_al_pacino.avi"
    face_detect_test(video_path)
