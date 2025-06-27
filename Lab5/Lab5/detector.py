from imageai.Detection import ObjectDetection
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
input_path = "./input/img4.png"
output_path = "./output/new_image.jpg"

# Створення директорії output якщо її немає
os.makedirs("./output", exist_ok=True)

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()  # Використовуємо повну версію YOLOv3

detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for item in detection:
    print(f"{item['name']}:{item['percentage_probability']}%")
