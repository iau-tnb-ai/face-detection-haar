import cv2
import os
import argparse

DEFAULT_OUTPUT_PATH = 'face-capture-images-fbf/'
HAARCASSCADE_FRONTFACE_ALT = 'haarcascades/haarcascade_frontalface_alt.xml'
HAARCASSCADE_EYE = 'haarcascades/haarcascade_eye.xml'
HAARCASSCADE_SMILE = 'haarcascades/haarcascade_smile.xml'


class VideoCapture:
    def __init__(self):
        self.count = 0
        self.argsObj = parse()
        self.faceCascade = cv2.CascadeClassifier(self.argsObj.input_path)
        self.videoSource = cv2.VideoCapture(0)

    def capture_frames(self):
        while True:

            frame_number = '%08d' % (self.count,)

            ret, frame = self.videoSource.read()

            screen_color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                screen_color,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            cv2.imshow('Detecting...', screen_color)

            if len(faces) == 0:
                pass

            elif len(faces) > 0:
                print('Face Detected')

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imwrite(DEFAULT_OUTPUT_PATH + frame_number + '.png', frame)

            self.count += 1

            if cv2.waitKey(1) == 27:
                break

        self.videoSource.release()
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        cv2.waitKey(500)


def parse():
    parser = argparse.ArgumentParser(description='Cascade path for face detection')
    parser.add_argument('-i', '--input_path', type=str, default=HAARCASSCADE_FRONTFACE_ALT, help='Cascade input path')
    parser.add_argument('-o', '--output_path', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Output path for pictures taken')
    args = parser.parse_args()
    return args


def clear_image_folder():
    if not (os.path.exists(DEFAULT_OUTPUT_PATH)):
        os.makedirs(DEFAULT_OUTPUT_PATH)

    else:
        for files in os.listdir(DEFAULT_OUTPUT_PATH):
            file_path = os.path.join(DEFAULT_OUTPUT_PATH, files)

            if os.path.isfile(file_path):
                os.unlink(file_path)

            else:
                continue


def main():
    clear_image_folder()
    fd_implement = VideoCapture()
    fd_implement.capture_frames()


if __name__ == '__main__':
    main()
