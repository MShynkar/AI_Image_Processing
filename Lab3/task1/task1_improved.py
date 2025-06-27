import cv2
import numpy as np

def lane_detection(image_path):
    """
    Функція для розпізнавання дорожньої розмітки на зображенні
    """
    # Завантаження зображення
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не вдалося завантажити зображення: {image_path}")
        return None
    
    original_img = img.copy()  # Зберігаємо оригінал для відображення
    
    # 1. Конвертація в сіре зображення
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Етап 1: Конвертація в сіре зображення завершена")
    
    # 2. Розмиття за фільтром Гаусса
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    print("Етап 2: Розмиття Гаусса завершене")
    
    # 3. Застосування алгоритму Кенні
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    print("Етап 3: Алгоритм Кенні застосований")
    
    # 4. Створення набору вершин для маски (адаптивно до розміру зображення)
    height, width = img.shape[:2]
    vertices = np.array([
        [(0, height),
         (int(width * 0.45), int(height * 0.6)), 
         (int(width * 0.55), int(height * 0.6)),
         (width, height)]
    ], dtype=np.int32)
    
    # Створення маски
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    print("Етап 4: Маска створена та застосована")
    
    # 5. Функція для відображення ліній
    def draw_lines(img, lines, color=[0, 255, 0], thickness=8):
        """Покращена функція для відображення ліній"""
        if lines is None:
            return
            
        x_bottom_pos = []
        x_upper_pos = []
        x_bottom_neg = []
        x_upper_neg = []

        y_bottom = height
        y_upper = int(height * 0.6)

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)

                # Фільтрація ліній за кутом нахилу
                if 0.3 < slope < 2:  # Права лінія
                    b = y1 - slope * x1
                    x_bottom_pos.append((y_bottom - b) / slope)
                    x_upper_pos.append((y_upper - b) / slope)
                elif -2 < slope < -0.3:  # Ліва лінія
                    b = y1 - slope * x1
                    x_bottom_neg.append((y_bottom - b) / slope)
                    x_upper_neg.append((y_upper - b) / slope)

        # Усереднення координат для стабільності
        lines_mean = []
        if x_bottom_pos and x_upper_pos:
            lines_mean.append([
                int(np.mean(x_bottom_pos)), y_bottom,
                int(np.mean(x_upper_pos)), y_upper
            ])
        if x_bottom_neg and x_upper_neg:
            lines_mean.append([
                int(np.mean(x_bottom_neg)), y_bottom,
                int(np.mean(x_upper_neg)), y_upper
            ])

        # Відображення ліній
        for line in lines_mean:
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)
        
        return len(lines_mean)

    # 6. Перетворення Хафа
    rho = 2
    theta = np.pi / 180
    threshold = 20
    min_line_length = 40
    max_line_gap = 20
    
    lines = cv2.HoughLinesP(
        masked_edges, rho, theta, threshold,
        np.array([]),
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    print("Етап 5: Перетворення Хафа завершене")
    
    # Відображення результатів
    lines_count = draw_lines(img, lines)
    print(f"Знайдено та відображено {lines_count} ліній")
    
    # Створення комбінованого зображення для демонстрації етапів
    combined = create_processing_stages_view(original_img, grayscale, blur, edges, masked_edges, img)
    
    return img, combined

def create_processing_stages_view(original, gray, blur, edges, masked_edges, result):
    """Створює зображення з усіма етапами обробки"""
    # Зменшуємо розмір для відображення
    scale = 0.3
    h, w = int(original.shape[0] * scale), int(original.shape[1] * scale)
    
    # Змінюємо розмір всіх зображень
    orig_small = cv2.resize(original, (w, h))
    gray_small = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (w, h))
    blur_small = cv2.resize(cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR), (w, h))
    edges_small = cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (w, h))
    masked_small = cv2.resize(cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR), (w, h))
    result_small = cv2.resize(result, (w, h))
    
    # Створюємо сітку 2x3
    top_row = np.hstack([orig_small, gray_small, blur_small])
    bottom_row = np.hstack([edges_small, masked_small, result_small])
    combined = np.vstack([top_row, bottom_row])
    
    # Додаємо підписи
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    
    labels = ["Original", "Grayscale", "Blur", "Edges", "Masked", "Result"]
    positions = [(10, 20), (w+10, 20), (2*w+10, 20), (10, h+40), (w+10, h+40), (2*w+10, h+40)]
    
    for label, pos in zip(labels, positions):
        cv2.putText(combined, label, pos, font, font_scale, color, thickness)
    
    return combined

if __name__ == "__main__":
    # Обробка зображення
    result_img, stages_img = lane_detection("img.jpg")
    
    if result_img is not None:
        # Відображення результату
        cv2.imshow("Lane Detection Result", result_img)
        cv2.imshow("Processing Stages", stages_img)
        
        print("\nНатисніть будь-яку клавішу для виходу...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Збереження результатів
        cv2.imwrite("result_lanes.jpg", result_img)
        cv2.imwrite("processing_stages.jpg", stages_img)
        print("Результати збережено: result_lanes.jpg, processing_stages.jpg")
