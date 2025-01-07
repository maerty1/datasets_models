import os
import cv2
import numpy as np
from ultralytics import YOLO
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from PIL import Image
import random
from tqdm import tqdm
import logging
import zipfile
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Отключаем логирование библиотеки ultralytics
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Путь к каталогу с исходными изображениями
SRC_IMAGE_DIR = "src_image"

# Путь к каталогу для сохранения датасета
DATASET_DIR = "dataset"

# Порог вероятности для успешной детекции 0.6 = 60%
CONFIDENCE_THRESHOLD = 0.9

# Создаем необходимые каталоги
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "valid", "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "valid", "labels"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "test", "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "test", "labels"), exist_ok=True)

# Загрузка модели YOLO для детекции объектов
model = YOLO("counter_water.pt", verbose=False)

# Словарь для хранения имен классов (ID: название класса)
class_names = {}

# Переменные для включения и отключения модификаторов
enable_horizontal_flip = False
enable_vertical_flip = False
enable_rotate = False
enable_brightness_contrast = False
enable_hue_saturation_value = False
enable_gauss_noise = False
enable_blur = False
enable_clahe = False
enable_random_gamma = False
enable_random_shadow = False

# Переменная для выбора классов
selected_classes = '*'  # '*' для выбора всех классов или список выбранных классов, например [0, 1, 2]

# Функция для сохранения изображения и данных
def save_image_and_data(image, name, bboxes, labels, save_dir):
    image_filename = f"{name}.jpg"
    txt_filename = f"{name}.txt"
    image_path = os.path.join(save_dir, "images", image_filename)
    txt_path = os.path.join(save_dir, "labels", txt_filename)

    # Сохраняем изображение
    cv2.imwrite(image_path, image)

    # Сохраняем данные в текстовый файл
    with open(txt_path, 'w') as f:
        for label, bbox in zip(labels, bboxes):
            # Преобразуем label в целое число
            f.write(f"{int(label)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

# Функция для загрузки изображения с помощью PIL
def load_image(image_path):
    try:
        # Загрузка изображения с помощью PIL
        image = Image.open(image_path)
        image = image.convert("RGB")  # Преобразование в RGB (если изображение в другом формате)
        return np.array(image)
    except Exception:
        return None

# Функция для изменения размера изображения
def resize_image(image, target_size=640):
    height, width, _ = image.shape
    if width > height:
        new_width = target_size
        new_height = int((target_size / width) * height)
    else:
        new_height = target_size
        new_width = int((target_size / height) * width)
    return cv2.resize(image, (new_width, new_height))

# Функция для детекции объектов и сохранения результатов
def detect_and_save(image_path, save_dir):
    # Преобразуем путь в строку без кодирования в байты
    image_path = os.path.normpath(image_path)  # Нормализуем путь

    image = load_image(image_path)
    if image is None:
        return None, None

    # Изменяем размер изображения
    image = resize_image(image)

    name = os.path.basename(image_path).split('.')[0]

    # Детекция объектов
    try:
        results = model(image)
    except Exception:
        return None, None

    # Получение координат bounding box и идентификаторов
    boxes = results[0].boxes
    height, width, _ = image.shape

    bboxes = []
    labels = []

    # Обработка детекций
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())

        if conf < CONFIDENCE_THRESHOLD:
            continue

        if selected_classes != '*' and cls not in selected_classes:
            continue

        x_center = (x1 + x2) / 2 / width
        y_center = (y1 + y2) / 2 / height
        bbox_width = (x2 - x1) / width
        bbox_height = (y2 - y1) / height

        if cls not in class_names:
            class_names[cls] = model.names[cls]

        bboxes.append([x_center, y_center, bbox_width, bbox_height])
        labels.append(cls)

    if len(bboxes) == 0:
        return None, None

    # Сохраняем изображение и данные
    save_image_and_data(image, name, bboxes, labels, save_dir)

    return bboxes, labels

# Функция для применения аугментации к изображению
def apply_augmentations(image, bboxes, labels):
    transforms = []

    if enable_horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if enable_vertical_flip:
        transforms.append(A.VerticalFlip(p=0.5))
    if enable_rotate:
        transforms.append(A.Rotate(limit=45, p=0.5))
    if enable_brightness_contrast:
        transforms.append(A.RandomBrightnessContrast(p=0.5))
    if enable_hue_saturation_value:
        transforms.append(A.HueSaturationValue(p=0.5))
    if enable_gauss_noise:
        transforms.append(A.GaussNoise(p=0.5))
    if enable_blur:
        transforms.append(A.Blur(p=0.5))
    if enable_clahe:
        transforms.append(A.CLAHE(p=0.5))
    if enable_random_gamma:
        transforms.append(A.RandomGamma(p=0.5))
    if enable_random_shadow:
        transforms.append(A.RandomShadow(p=0.5))

    if not transforms:
        return image, bboxes, labels

    transform = A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    augmented = transform(image=image, bboxes=bboxes, labels=labels)
    return augmented['image'], augmented['bboxes'], augmented['labels']

# Обработка всех изображений в каталоге
image_files = [f for f in os.listdir(SRC_IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

train_size = int(0.7 * len(image_files))
valid_size = int(0.2 * len(image_files))
test_size = len(image_files) - train_size - valid_size

train_files = image_files[:train_size]
valid_files = image_files[train_size:train_size + valid_size]
test_files = image_files[train_size + valid_size:]

# Обработка изображений с прогресс-баром
for filename in tqdm(train_files, desc="Processing train images"):
    image_path = os.path.join(SRC_IMAGE_DIR, filename)
    bboxes, labels = detect_and_save(image_path, os.path.join(DATASET_DIR, "train"))
    if bboxes is not None:
        image = load_image(image_path)
        if image is not None:
            augmented_image, augmented_bboxes, augmented_labels = apply_augmentations(image, bboxes, labels)
            if augmented_image is not image:
                save_image_and_data(augmented_image, f"{filename.split('.')[0]}_augmented", augmented_bboxes, augmented_labels, os.path.join(DATASET_DIR, "train"))

for filename in tqdm(valid_files, desc="Processing valid images"):
    image_path = os.path.join(SRC_IMAGE_DIR, filename)
    bboxes, labels = detect_and_save(image_path, os.path.join(DATASET_DIR, "valid"))
    if bboxes is not None:
        image = load_image(image_path)
        if image is not None:
            augmented_image, augmented_bboxes, augmented_labels = apply_augmentations(image, bboxes, labels)
            if augmented_image is not image:
                save_image_and_data(augmented_image, f"{filename.split('.')[0]}_augmented", augmented_bboxes, augmented_labels, os.path.join(DATASET_DIR, "valid"))

for filename in tqdm(test_files, desc="Processing test images"):
    image_path = os.path.join(SRC_IMAGE_DIR, filename)
    bboxes, labels = detect_and_save(image_path, os.path.join(DATASET_DIR, "test"))
    if bboxes is not None:
        image = load_image(image_path)
        if image is not None:
            augmented_image, augmented_bboxes, augmented_labels = apply_augmentations(image, bboxes, labels)
            if augmented_image is not image:
                save_image_and_data(augmented_image, f"{filename.split('.')[0]}_augmented", augmented_bboxes, augmented_labels, os.path.join(DATASET_DIR, "test"))

# Создаем единый словарь классов
combined_class_names = {}
current_index = 0

# Добавляем классы из model
for cls_id, cls_name in model.names.items():
    combined_class_names[current_index] = cls_name
    current_index += 1

# Сортируем классы по ID
sorted_class_names = dict(sorted(combined_class_names.items()))

# Создаем файл dataset.yaml
dataset_yaml = CommentedMap([
    ("train", "train/images"),
    ("val", "valid/images"),
    ("test", "test/images")
])

# Добавляем блок names с отсортированными классами
dataset_yaml["names"] = sorted_class_names

# Добавляем пустую строку перед блоком names
dataset_yaml.yaml_set_comment_before_after_key("names", before="\n")

# Используем ruamel.yaml для сохранения порядка ключей
yaml = YAML()
yaml.default_flow_style = False

with open(os.path.join(DATASET_DIR, "data.yaml"), 'w') as f:
    yaml.dump(dataset_yaml, f)

print("Процесс завершён. Результаты сохранены в каталоге:", DATASET_DIR)

# Функция для упаковки датасета в ZIP-архив
def zip_dataset(dataset_dir, zip_file):
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dataset_dir)
                zipf.write(file_path, arcname)

# Упаковка датасета в ZIP-архив
zip_dataset(DATASET_DIR, "dataset.zip")

print(f"Датасет упакован в архив: dataset.zip")
