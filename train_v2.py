import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'ultralytics'))

from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules.head import Segment
import logging
import traceback

logging.basicConfig(level=logging.DEBUG)

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # Пути
custom_cfg_path = '/home/asad/Classification/YOLO_GRAVEL_train/ultralytics_fine_tune reg_max/ultralytics/ultralytics/cfg/models/v9/yolov9e-seg.yaml'
pretrained_ckpt_path = 'yolov9e-seg.pt'
#ata_yaml = '/home/asad/Classification/YOLO_GRAVEL_train/GRAVEL/data.yaml'
data_yaml = "/home/asad/stone.v14i.yolov11/GLOBAL_DATA/data.yaml"
#data_yaml = "/home/asad/stone.v14i.yolov11/data.yaml"
# 1. Загружаем модель (без обучения) и заменяем голову
model = SegmentationModel(cfg=custom_cfg_path, ch=3, nc=1)

torch.cuda.empty_cache()


# 2. Меняем голову с новым reg_max
if hasattr(model.model[-1], 'reg_max'):
    print(f"Original reg_max: {model.model[-1].reg_max}")

    old_head = model.model[-1]
    new_head = Segment(
        nc=old_head.nc,
        ch= [m[0].conv.in_channels for m in old_head.cv4] , 
        reg_max=1
    )
    model.model[-1] = new_head
    print(f"New reg_max: {model.model[-1].reg_max}")

# 3. Загружаем веса из предобученной модели
if os.path.exists(pretrained_ckpt_path):
    pretrained = torch.load(pretrained_ckpt_path, map_location='cpu')
    pretrained_state_dict = pretrained['model'].float().state_dict()

    current_state_dict = model.state_dict()
    for name, param in pretrained_state_dict.items():
        if name in current_state_dict and current_state_dict[name].shape == param.shape:
            current_state_dict[name] = param

    model.load_state_dict(current_state_dict, strict=False)
    print("Предобученные веса загружены.")

# 4. Встраиваем модель обратно в YOLO для обучения
yolo_model = YOLO(custom_cfg_path)
yolo_model.model = model  # заменяем на кастомную модель с reg_max=64

# 5. Тренировка
try:
    results = yolo_model.train(
        data=data_yaml,
        task='segment',
        epochs=1000,
        imgsz=640,
        batch=9,
        mask_ratio=4,
        device=[0, 1, 2],
        augment=True,
        cos_lr=True,
        max_det=300,
        exist_ok=False,
        overlap_mask=False,
        val=True, 
        patience = 200,
        freeze='backbone',
        close_mosaic=950,
        #workers=12
    )
except Exception as e:
    logging.error("Ошибка при обучении модели:")
    logging.error(str(e))
    logging.error(traceback.format_exc())

# # 6. Сохраняем модель полностью
save_path = 'custom_yolov11x_regmax1_trained_BIG_DATA.pt'
#model = SegmentationModel(cfg=custom_cfg_path, ch=3, nc=1)
yolo_model.save(save_path)
#print(f"Модель сохранена в {model}")
print(f"Итоговый reg_max: {model.model[-1].reg_max}")
