import cv2
import numpy as np

# Ініціалізація HOG детектора для виявлення людей
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Ініціалізація каскадного класифікатора для виявлення облич
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Перевірка завантаження класифікатора
if face_cascade.empty():
    print("Помилка: Не вдалося завантажити каскадний класифікатор для облич!")
    exit()

# Завантаження відео
cap = cv2.VideoCapture("vid.mp4")

# Перевірка, чи завантажилось відео
if not cap.isOpened():
    print("Помилка: Не вдалося завантажити відео 'vid.mp4'!")
    exit()

# Отримуємо інформацію про відео
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"Інформація про відео:")
print(f"FPS: {fps}")
print(f"Кількість кадрів: {frame_count}")
print(f"Тривалість: {duration:.2f} секунд")
print("Натисніть 'q' для виходу, 'p' для паузи/відновлення")

cv2.startWindowThread()

frame_number = 0
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        frame_number += 1
        
        if not ret:
            print("Кінець відео або помилка читання кадру")
            break

        # Зміна розміру кадру для кращої продуктивності
        frame = cv2.resize(frame, (800, 500))
        gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Виявлення людей за допомогою HOG
        boxes, weights = hog.detectMultiScale(gray_filter, winStride=(8, 8), padding=(4, 4), scale=1.05)
        
        # Застосування Non-Maximum Suppression для усунення дублікатів
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        
        if len(boxes) > 0:
            # Застосовуємо NMS
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), weights.flatten(), 0.5, 0.4)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    (xa, ya, xb, yb) = boxes[i]
                    # Малюємо жовтий прямокутник навколо пішоходів
                    cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 255), 2)
                    cv2.putText(frame, 'Person', (xa, ya-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Виявлення облич за допомогою каскадів Хаара
        face_rects = face_cascade.detectMultiScale(gray_filter, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in face_rects:
            # Малюємо синій прямокутник навколо облич
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Додаємо інформацію на кадр
        cv2.putText(frame, f'Frame: {frame_number}/{frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'People: {len(indices) if len(boxes) > 0 and len(indices) > 0 else 0}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Faces: {len(face_rects)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, 'Q-quit, P-pause/resume', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Pedestrian and Face Detection - Task 3', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        print("Пауза" if paused else "Відновлення")

cap.release()
cv2.destroyAllWindows()
