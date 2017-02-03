import cv2

face_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_smile.xml")

videoCapture = cv2.VideoCapture(0)

while True:
    ret, frame = videoCapture.read()
    screenColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        screenColor,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0), 2)
        roi_gray = screenColor[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + ew), (0, 255, 0), 2)

    cv2.imshow("Detecting...", frame)
    if cv2.waitKey(30) == 27:
        break

videoCapture.release()
cv2.waitKey(500)
cv2.destroyAllWindows()
cv2.waitKey(500)
