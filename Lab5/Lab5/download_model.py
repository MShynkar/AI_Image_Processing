import os
import urllib.request
from pathlib import Path

def download_model():
    """Завантаження YOLOv3 моделі для imageai"""
    
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "yolov3.pt"
    
    if not model_path.exists():
        print("Завантаження моделі YOLOv3...")
        url = "https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt"
        
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Модель збережена в {model_path}")
        except Exception as e:
            print(f"Помилка завантаження моделі: {e}")
            print("Будь ласка, завантажте модель вручну з:")
            print(url)
    else:
        print(f"Модель вже існує: {model_path}")

if __name__ == "__main__":
    download_model()
