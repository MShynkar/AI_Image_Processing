import cv2

# Ініціалізація каскадних класифікаторів
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Перевірка, чи завантажились каскади
if face_cascade.empty() or smile_cascade.empty() or eye_cascade.empty():
    print("Помилка: Не вдалося завантажити каскадні класифікатори!")
    exit()

scaling_factor = 0.5
frame = cv2.imread('image.jpg')

# Перевірка, чи завантажилось зображення
if frame is None:
    print("Помилка: Не вдалося завантажити зображення 'image.jpg'!")
    exit()

gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Виявлення облич з покращеними параметрами
face_rects = face_cascade.detectMultiScale(gray_filter, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
total_smiles = 0
total_eyes = 0

for (x, y, w, h) in face_rects:
    # Малюємо прямокутник навколо обличчя (синій колір)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Виділяємо область обличчя для пошуку очей та посмішок
    roi_gray = gray_filter[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]
    
    # Виявлення посмішок з покращеними параметрами
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
    
    # Виявлення очей з покращеними параметрами
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

    # Малюємо прямокутники навколо посмішок (зелений колір)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        total_smiles += 1
    
    # Малюємо прямокутники навколо очей (червоний колір)  
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
        total_eyes += 1

print(f"Результати виявлення:")
print(f"Знайдено {len(face_rects)} облич(чя)")
print(f"Знайдено {total_smiles} посмішок")
print(f"Знайдено {total_eyes} очей")

# Додаємо текст на зображення
cv2.putText(frame, f'Faces: {len(face_rects)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(frame, f'Smiles: {total_smiles}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(frame, f'Eyes: {total_eyes}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow('Face Detection - Task 1', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
