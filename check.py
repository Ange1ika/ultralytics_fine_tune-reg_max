import sys
import os
import torch
import cv2
import numpy as np

# Добавление пути к ultralytics
sys.path.append(os.path.join(os.path.dirname(__file__), 'ultralytics'))

from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules.head import Segment

# Пути
custom_cfg_path = '/home/asad/Classification/YOLO_GRAVEL_train/ultralytics_fine_tune reg_max/ultralytics/ultralytics/cfg/models/11/yolo11x-seg.yaml'
model_path = '/home/asad/Classification/YOLO_GRAVEL_train/runs/segment/train34/weights/best.pt'
folder_path = '/home/asad/Downloads/pipeline_tests_v.2.0/Data/sample/big_test'

# Загрузка модели
model = YOLO(model_path)
print("reg_max:", model.model.model[-1].reg_max)

# Инференс и визуализация
def infer_and_visualize(image_path):
    img = cv2.imread(image_path)

    # results = model.predict(
    #     img,
    #     conf=0.1,
    #     imgsz=640,
    #     max_det=300,
    #     retina_masks=False,
    #     show_boxes=False,
    #     verbose=False,
    #     overlap_mask=False,
    #     iou=0.7,
    #     rect = True,

    # )
    results = model(img)

    result = results[0]

    output_mask_dir = 'predicted_masks'
    os.makedirs(output_mask_dir, exist_ok=True)


    # Отрисовка масок
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # [N, H, W]
        orig_h, orig_w = img.shape[:2]

    #     for i, mask in enumerate(masks):
    #         color = (0, 255, 0)
    #         #mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    #         mask_uint8 = (masks * 255).astype(np.uint8)
    #         contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         contours = cv2.resize(contours, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    #         cv2.drawContours(img, contours, -1, color, 2)

    # # Сохранение результата
    # save_path = os.path.join("output", os.path.basename(image_path))
    # os.makedirs("output", exist_ok=True)
    # cv2.imwrite(save_path, img)
    # print(f"Saved to {save_path}")


        for i, mask in enumerate(masks):
            color = (0, 255, 0)

            # 1. Масштабируем маску до оригинального размера (маска типа float32: 0..1)
            mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # 2. Преобразуем в uint8 (0..255) для бинаризации
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            # # 3. Применяем адаптивную бинаризацию
            # binary_mask = cv2.adaptiveThreshold(
            #     mask_uint8, 
            #     255,
            #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # или cv2.ADAPTIVE_THRESH_MEAN_C
            #     cv2.THRESH_BINARY,
            #     11,  # размер окна (должен быть нечётным)
            #     2    # константа, которая вычитается
            # )

            # 4. Сохраняем бинарную маску

            # 5. Поиск контуров (точный)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # 6. Отрисовка контуров на оригинальном изображении
            cv2.drawContours(img, contours, -1, color, 2)
            
        # Сохранение результата
    save_path = os.path.join("output", os.path.basename(image_path))
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"Saved to {save_path}")

# Обработка папки
def process_folder_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_file}...")
        infer_and_visualize(image_path)

# Запуск
process_folder_images(folder_path)
