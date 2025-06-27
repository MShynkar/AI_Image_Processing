import cv2
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_lane_detection_steps(image_path):
    """
    Демонстрація всіх етапів розпізнавання дорожньої розмітки
    згідно з теоретичними відомостями лабораторної роботи
    """
    
    # Завантаження зображення
    img = cv2.imread(image_path)
    if img is None:
        print(f"Помилка: не вдалося завантажити зображення {image_path}")
        return
    
    # Конвертація з BGR в RGB для правильного відображення в matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("=== ДЕМОНСТРАЦІЯ ЕТАПІВ РОЗПІЗНАВАННЯ ДОРОЖНЬОЇ РОЗМІТКИ ===")
    
    # Етап 1: Конвертація в сіре зображення
    print("Етап 1: Конвертація в сіре зображення")
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Етап 2: Розмиття за фільтром Гаусса
    print("Етап 2: Застосування розмиття Гаусса")
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    
    # Етап 3: Застосування алгоритму Кенні
    print("Етап 3: Застосування алгоритму Кенні для виявлення країв")
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    
    # Етап 4: Створення маски
    print("Етап 4: Створення маски для виділення області інтересу")
    height, width = img.shape[:2]
    vertices = np.array([
        [(0, height),
         (int(width * 0.45), int(height * 0.6)), 
         (int(width * 0.55), int(height * 0.6)),
         (width, height)]
    ], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Етап 5: Перетворення Хафа
    print("Етап 5: Застосування перетворення Хафа для виявлення ліній")
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
    
    # Створення зображення з усіма виявленими лініями
    lines_img = img.copy()
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Етап 6: Усереднення та відображення фінальних ліній
    print("Етап 6: Усереднення ліній та створення фінального результату")
    final_img = img.copy()
    
    def draw_averaged_lines(img, lines, color=[0, 255, 0], thickness=8):
        if lines is None:
            return 0
            
        left_lines = []
        right_lines = []
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                if 0.3 < slope < 2:
                    right_lines.append((slope, intercept))
                elif -2 < slope < -0.3:
                    left_lines.append((slope, intercept))
        
        y_bottom = height
        y_top = int(height * 0.6)
        lines_drawn = 0
        
        if left_lines:
            avg_slope = np.mean([line[0] for line in left_lines])
            avg_intercept = np.mean([line[1] for line in left_lines])
            x_bottom = int((y_bottom - avg_intercept) / avg_slope)
            x_top = int((y_top - avg_intercept) / avg_slope)
            cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), color, thickness)
            lines_drawn += 1
        
        if right_lines:
            avg_slope = np.mean([line[0] for line in right_lines])
            avg_intercept = np.mean([line[1] for line in right_lines])
            x_bottom = int((y_bottom - avg_intercept) / avg_slope)
            x_top = int((y_top - avg_intercept) / avg_slope)
            cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), color, thickness)
            lines_drawn += 1
            
        return lines_drawn
    
    lines_count = draw_averaged_lines(final_img, lines)
    print(f"Виявлено та відображено {lines_count} дорожніх ліній")
    
    # Створення великого зображення з усіма етапами
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Етапи розпізнавання дорожньої розмітки', fontsize=16)
    
    # Оригінальне зображення
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('1. Оригінальне зображення')
    axes[0, 0].axis('off')
    
    # Сіре зображення
    axes[0, 1].imshow(grayscale, cmap='gray')
    axes[0, 1].set_title('2. Сіре зображення')
    axes[0, 1].axis('off')
    
    # Розмите зображення
    axes[0, 2].imshow(blur, cmap='gray')
    axes[0, 2].set_title('3. Розмиття Гаусса')
    axes[0, 2].axis('off')
    
    # Краї
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('4. Алгоритм Кенні')
    axes[1, 0].axis('off')
    
    # Маска
    axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title('5. Маска області інтересу')
    axes[1, 1].axis('off')
    
    # Замасковані краї
    axes[1, 2].imshow(masked_edges, cmap='gray')
    axes[1, 2].set_title('6. Замасковані краї')
    axes[1, 2].axis('off')
    
    # Всі лінії Хафа
    axes[2, 0].imshow(cv2.cvtColor(lines_img, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('7. Лінії Хафа (всі)')
    axes[2, 0].axis('off')
    
    # Фінальний результат
    axes[2, 1].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title('8. Фінальний результат')
    axes[2, 1].axis('off')
    
    # Порівняння до/після
    comparison = np.hstack([img_rgb, cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)])
    axes[2, 2].imshow(comparison)
    axes[2, 2].set_title('9. До/Після')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('lane_detection_stages.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Збереження результатів
    cv2.imwrite('result_final.jpg', final_img)
    cv2.imwrite('result_all_lines.jpg', lines_img)
    
    print("\n=== РЕЗУЛЬТАТИ ===")
    print("✅ Всі етапи обробки виконані успішно")
    print("✅ Графік збережено як: lane_detection_stages.png")
    print("✅ Фінальний результат збережено як: result_final.jpg")
    print("✅ Всі лінії Хафа збережено як: result_all_lines.jpg")
    
    return final_img

def analyze_parameters():
    """Аналіз впливу різних параметрів на результат"""
    print("\n=== АНАЛІЗ ПАРАМЕТРІВ ===")
    print("Параметри алгоритму Кенні:")
    print("- Нижній поріг: 50 (видаляє слабкі краї)")
    print("- Верхній поріг: 150 (зберігає сильні краї)")
    print("\nПараметри розмиття Гаусса:")
    print("- Розмір ядра: 5x5 (усуває шум)")
    print("\nПараметри перетворення Хафа:")
    print("- rho: 2 (точність відстані в пікселях)")
    print("- theta: π/180 (точність кута в радіанах)")
    print("- threshold: 20 (мінімальна кількість перетинів)")
    print("- min_line_length: 40 (мінімальна довжина лінії)")
    print("- max_line_gap: 20 (максимальний розрив між сегментами)")

if __name__ == "__main__":
    # Демонстрація для зображення
    print("Запуск демонстрації розпізнавання дорожньої розмітки...")
    
    try:
        result = demonstrate_lane_detection_steps("img.jpg")
        analyze_parameters()
        
        print("\nДемонстрація завершена успішно!")
        print("Перевірте збережені файли результатів.")
        
    except Exception as e:
        print(f"Помилка під час виконання: {e}")
        print("Переконайтеся, що файл img.jpg існує в поточній директорії")
