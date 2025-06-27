import cv2
import numpy as np
import time

class LaneDetector:
    """Клас для розпізнавання дорожньої розмітки"""
    
    def __init__(self):
        # Параметри для обробки зображення
        self.kernel_size = 5
        self.low_threshold = 50
        self.high_threshold = 150
        
        # Параметри для перетворення Хафа
        self.rho = 2
        self.theta = np.pi / 180
        self.threshold = 20
        self.min_line_length = 40
        self.max_line_gap = 20
        
        # Статистика
        self.frame_count = 0
        self.start_time = time.time()
        
    def create_mask(self, img):
        """Створює маску для виділення області дороги"""
        height, width = img.shape[:2]
        vertices = np.array([
            [(0, height),
             (int(width * 0.4), int(height * 0.6)), 
             (int(width * 0.6), int(height * 0.6)),
             (width, height)]
        ], dtype=np.int32)
        
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        return mask
    
    def draw_lines(self, img, lines, color=[0, 255, 0], thickness=8):
        """Покращена функція для відображення ліній з усередненням"""
        if lines is None:
            return 0
            
        height = img.shape[0]
        left_lines = []
        right_lines = []
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Класифікація ліній за нахилом
                if 0.3 < slope < 2:  # Права лінія
                    right_lines.append((slope, intercept))
                elif -2 < slope < -0.3:  # Ліва лінія
                    left_lines.append((slope, intercept))
        
        # Усереднення параметрів ліній
        y_bottom = height
        y_top = int(height * 0.6)
        
        lines_drawn = 0
        
        # Відображення лівої лінії
        if left_lines:
            avg_slope = np.mean([line[0] for line in left_lines])
            avg_intercept = np.mean([line[1] for line in left_lines])
            
            x_bottom = int((y_bottom - avg_intercept) / avg_slope)
            x_top = int((y_top - avg_intercept) / avg_slope)
            
            cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), color, thickness)
            lines_drawn += 1
        
        # Відображення правої лінії
        if right_lines:
            avg_slope = np.mean([line[0] for line in right_lines])
            avg_intercept = np.mean([line[1] for line in right_lines])
            
            x_bottom = int((y_bottom - avg_intercept) / avg_slope)
            x_top = int((y_top - avg_intercept) / avg_slope)
            
            cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), color, thickness)
            lines_drawn += 1
            
        return lines_drawn
    
    def process_frame(self, frame):
        """Обробка одного фрейму"""
        # 1. Конвертація в сіре зображення
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Розмиття Гаусса
        blur = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)
        
        # 3. Виявлення країв за алгоритмом Кенні
        edges = cv2.Canny(blur, self.low_threshold, self.high_threshold)
        
        # 4. Застосування маски
        mask = self.create_mask(edges)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 5. Виявлення ліній за допомогою перетворення Хафа
        lines = cv2.HoughLinesP(
            masked_edges, 
            self.rho, 
            self.theta, 
            self.threshold,
            np.array([]),
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # 6. Відображення ліній
        lines_count = self.draw_lines(frame, lines)
        
        # Оновлення статистики
        self.frame_count += 1
        
        return frame, lines_count
    
    def get_fps(self):
        """Розрахунок FPS"""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0

def process_video(video_path, show_debug=False):
    """Функція для обробки відео з покращеннями"""
    
    # Ініціалізація детектора
    detector = LaneDetector()
    
    # Відкриття відео
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити відео {video_path}")
        return
    
    # Отримання інформації про відео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Відео: {width}x{height}, FPS: {fps:.2f}, Фреймів: {total_frames}")
    print("Натисніть 'q' для виходу, 'p' для паузи, 'd' для перемикання режиму debug")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Кінець відео або помилка читання")
                break
        
        # Обробка фрейму
        processed_frame, lines_count = detector.process_frame(frame.copy())
        
        # Додавання інформації на екран
        fps_current = detector.get_fps()
        cv2.putText(processed_frame, f"FPS: {fps_current:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Lines: {lines_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Frame: {detector.frame_count}/{total_frames}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Відображення результату
        if show_debug:
            # Режим debug - показуємо етапи обробки
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (detector.kernel_size, detector.kernel_size), 0)
            edges = cv2.Canny(blur, detector.low_threshold, detector.high_threshold)
            
            # Створюємо сітку для відображення
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            blur_bgr = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Зменшуємо розмір для відображення
            scale = 0.4
            h, w = int(height * scale), int(width * scale)
            
            frame_small = cv2.resize(frame, (w, h))
            gray_small = cv2.resize(gray_bgr, (w, h))
            edges_small = cv2.resize(edges_bgr, (w, h))
            result_small = cv2.resize(processed_frame, (w, h))
            
            # Створюємо сітку 2x2
            top_row = np.hstack([frame_small, gray_small])
            bottom_row = np.hstack([edges_small, result_small])
            debug_view = np.vstack([top_row, bottom_row])
            
            # Додаємо підписи
            cv2.putText(debug_view, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_view, "Grayscale", (w+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_view, "Edges", (10, h+45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_view, "Result", (w+10, h+45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Lane Detection - Debug Mode', debug_view)
        else:
            cv2.imshow('Lane Detection', processed_frame)
        
        # Обробка клавіш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Пауза" if paused else "Відтворення")
        elif key == ord('d'):
            show_debug = not show_debug
            print("Debug режим:", "увімкнено" if show_debug else "вимкнено")
    
    # Фінальна статистика
    final_fps = detector.get_fps()
    print(f"\nСтатистика:")
    print(f"Оброблено фреймів: {detector.frame_count}")
    print(f"Середній FPS: {final_fps:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Запуск обробки відео
    video_path = "vid.mp4"
    process_video(video_path, show_debug=False)
