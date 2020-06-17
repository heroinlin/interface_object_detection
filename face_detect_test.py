import cv2
from face_detect_mtcnn import FaceDetector, show_bboxes


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
        bounding_boxes, landmarks = face_detector.detect(img)
        image = show_bboxes(img, bounding_boxes, landmarks)

        # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow('0', image)
        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    video_path = r"F:\Database\face_videos\0188_03_021_al_pacino.avi"
    face_detect_test(video_path)
