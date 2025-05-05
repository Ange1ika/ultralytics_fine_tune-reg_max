import os

# Пути к папкам с изображениями и аннотациями
images_dir = '/home/asad/stone.v14i.yolov11/GLOBAL_DATA/valid/images'
labels_dir = '/home/asad/stone.v14i.yolov11/GLOBAL_DATA/valid/labels'

# Поддерживаемые расширения изображений
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# Список изображений
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

# Проверка наличия аннотаций
missing_annotations = []

for image_file in image_files:
    base_name = os.path.splitext(image_file)[0]
    label_file = base_name + '.txt'
    label_path = os.path.join(labels_dir, label_file)
    
    if not os.path.exists(label_path):
        missing_annotations.append(image_file)

# Вывод результата
if missing_annotations:
    print("Не найдены аннотации для следующих изображений:")
    for f in missing_annotations:
        print(f"- {f}")
else:
    print("Аннотации найдены для всех изображений.")