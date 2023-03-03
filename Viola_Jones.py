import cv2

#инициализация поиск лица (по умолчанию каскад Хаара)
face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

#загрузка изображения
image = cv2.resize(cv2.imread('eyes.jpg'), (0, 0), fx=0.5, fy=0.5)

#преобразование изображения к оттенкам серого
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# обнаружение всех лиц на изображении
faces = face_cascade_db.detectMultiScale(image_gray, 2.7, 5)

# построение рамки для всех обнаруженных лиц
for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

#вывод изображения с обнаруженными лицами
cv2.imshow('rez', image)
cv2.waitKey()
