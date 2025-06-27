import cv2

# Ініціалізація каскадних класифікаторів
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Перевірка, чи завантажились каскади
if face_cascade.empty() or smile_cascade.empty() or eye_cascade.empty():
    print("Помилка: Не вдалося завантажити каскадні класифікатори!")
    exit()

# Ініціалізація камери
cap = cv2.VideoCapture(0)

# Перевірка, чи підключена камера
if not cap.isOpened():
    print("Помилка: Не вдалося підключитись до камери!")
    exit()

print("Натисніть 'q' для виходу, 's' для збереження кадру")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Помилка: Не вдалося прочитати кадр з камери!")
        break

    # Конвертуємо в градації сірого
    gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Виявлення облич з покращеними параметрами
    face_rects = face_cascade.detectMultiScale(gray_filter, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    total_smiles = 0
    total_eyes = 0
    
    for (x, y, w, h) in face_rects:
        # Малюємо прямокутник навколо обличчя (синій колір)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Виділяємо область обличчя
        roi_gray = gray_filter[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Виявлення посмішок з покращеними параметрами
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(30, 30))
        # Виявлення очей з покращеними параметрами
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        # Малюємо прямокутники навколо посмішок (зелений колір)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            total_smiles += 1

        # Малюємо прямокутники навколо очей (червоний колір)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            total_eyes += 1
    
    # Додаємо інформацію на екран
    cv2.putText(frame, f'Faces: {len(face_rects)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Smiles: {total_smiles}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Eyes: {total_eyes}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'Press Q to quit, S to save frame', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Real-time Face Detection - Task 2', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('saved_frame.jpg', frame)
        print("Кадр збережено як 'saved_frame.jpg'")

cap.release()
cv2.destroyAllWindows()
