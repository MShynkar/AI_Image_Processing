from imageai.Detection import VideoObjectDetection
import torch
import os

# Налаштування PyTorch для роботи з GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Обмеження пам'яті GPU (опціонально)
    torch.cuda.set_per_process_memory_fraction(0.5)
else:
    device = torch.device("cpu")
    print("Using CPU")

# Встановлення змінної середовища для оптимізації
os.environ['TORCH_HOME'] = './models'


model_path = "./models/yolov3.pt"  # Змінено на PyTorch модель
input_path = "./input/vid.mp4"
output_path = "./output/detected_vid.mp4"  # Додано розширення .mp4

# Створення директорії output якщо її немає
os.makedirs("./output", exist_ok=True)

# Перевірка існування вхідного файлу
if not os.path.exists(input_path):
    print(f"Помилка: Файл {input_path} не знайдено!")
    exit(1)

print("Ініціалізація детектора відео...")
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()  # Використовуємо повну версію YOLOv3

print("Завантаження моделі...")
detector.setModelPath(model_path)
detector.loadModel()

print("Початок обробки відео...")
print("Це може зайняти декілька хвилин...")

# Налаштування для кращої сумісності з MP4
detection = detector.detectObjectsFromVideo(
    input_file_path=input_path, 
    output_file_path=output_path,
    frames_per_second=15,  # Знижено FPS для стабільності
    minimum_percentage_probability=50,  # Мінімальна ймовірність детекції
    log_progress=True  # Виводити прогрес
)

print(f"Обробка завершена! Результат збережено в: {output_path}")
