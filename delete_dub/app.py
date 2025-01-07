import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from tqdm import tqdm
import zipfile
import random
import shutil

# Путь к каталогу с датасетом
DATASET_DIR = "dataset"

# Переменные для управления выполнением функций
REMOVE_DUPLICATES = True  # Удаление дубликатов изображений
REMOVE_EMPTY_LABELS = True  # Удаление пустых меток
REDISTRIBUTE_FILES = True  # Перераспределение файлов
ZIP_DATASET = True  # Упаковка датасета в ZIP-архив

# Загрузка предобученной модели ResNet
model = models.resnet50(pretrained=True)
model.eval()

# Преобразование изображений для входа в модель
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображения
    transforms.ToTensor(),  # Преобразование изображения в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Нормализация изображения
])

# Функция для извлечения признаков изображения
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')  # Открытие и преобразование изображения в RGB
    image = transform(image).unsqueeze(0)  # Применение преобразований и добавление размерности
    with torch.no_grad():  # Отключение градиентов
        features = model(image)  # Извлечение признаков
    return features.numpy().flatten()  # Преобразование признаков в одномерный массив

# Функция для удаления дубликатов
def remove_duplicates(temp_image_dir, temp_label_dir):
    # Словарь для хранения признаков изображений
    image_features = {}

    # Проходим по всем файлам во временном каталоге с изображениями
    for filename in tqdm(os.listdir(temp_image_dir), desc=f"Обработка {temp_image_dir}"):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(temp_image_dir, filename)
            features = extract_features(image_path)

            # Проверяем наличие дубликатов
            for existing_features in image_features.values():
                if np.linalg.norm(features - existing_features) < 0.1:  # Порог схожести
                    # Удаляем дубликат изображения
                    os.remove(image_path)
                    print(f"Удалено дублирующееся изображение: {image_path}")

                    # Удаляем соответствующую метку
                    label_filename = f"{os.path.splitext(filename)[0]}.txt"
                    label_path = os.path.join(temp_label_dir, label_filename)
                    if os.path.exists(label_path):
                        os.remove(label_path)
                        print(f"Удалена соответствующая метка: {label_path}")
                    break
            else:
                image_features[image_path] = features

# Функция для удаления пустых меток
def remove_empty_labels(temp_label_dir, temp_image_dir):
    # Проходим по всем файлам во временном каталоге с метками
    for label_filename in tqdm(os.listdir(temp_label_dir), desc=f"Обработка {temp_label_dir}"):
        if label_filename.endswith('.txt'):
            label_path = os.path.join(temp_label_dir, label_filename)
            with open(label_path, 'r') as file:
                content = file.read().strip()
                if not content:
                    # Удаляем пустую метку
                    os.remove(label_path)
                    print(f"Удалена пустая метка: {label_path}")

                    # Удаляем соответствующее изображение
                    image_filename = f"{os.path.splitext(label_filename)[0]}.jpg"
                    image_path = os.path.join(temp_image_dir, image_filename)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"Удалено соответствующее изображение: {image_path}")

# Функция для перераспределения файлов
def redistribute_files(dataset_dir, temp_image_dir, temp_label_dir):
    # Создаем список всех файлов
    image_files = [f for f in os.listdir(temp_image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    label_files = [f for f in os.listdir(temp_label_dir) if f.endswith('.txt')]

    # Перемешиваем список
    random.seed(42)
    random.shuffle(image_files)

    # Разделяем список на три части
    train_size = int(0.7 * len(image_files))
    valid_size = int(0.2 * len(image_files))
    test_size = len(image_files) - train_size - valid_size

    train_images = image_files[:train_size]
    valid_images = image_files[train_size:train_size + valid_size]
    test_images = image_files[train_size + valid_size:]

    # Создаем каталоги для обучения, валидации и тестирования
    train_image_dir = os.path.join(dataset_dir, "train", "images")
    valid_image_dir = os.path.join(dataset_dir, "valid", "images")
    test_image_dir = os.path.join(dataset_dir, "test", "images")
    train_label_dir = os.path.join(dataset_dir, "train", "labels")
    valid_label_dir = os.path.join(dataset_dir, "valid", "labels")
    test_label_dir = os.path.join(dataset_dir, "test", "labels")

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(valid_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # Перемещаем файлы в соответствующие каталоги
    for image_file in train_images:
        shutil.move(os.path.join(temp_image_dir, image_file), os.path.join(train_image_dir, image_file))
        label_file = f"{os.path.splitext(image_file)[0]}.txt"
        if label_file in label_files:
            shutil.move(os.path.join(temp_label_dir, label_file), os.path.join(train_label_dir, label_file))

    for image_file in valid_images:
        shutil.move(os.path.join(temp_image_dir, image_file), os.path.join(valid_image_dir, image_file))
        label_file = f"{os.path.splitext(image_file)[0]}.txt"
        if label_file in label_files:
            shutil.move(os.path.join(temp_label_dir, label_file), os.path.join(valid_label_dir, label_file))

    for image_file in test_images:
        shutil.move(os.path.join(temp_image_dir, image_file), os.path.join(test_image_dir, image_file))
        label_file = f"{os.path.splitext(image_file)[0]}.txt"
        if label_file in label_files:
            shutil.move(os.path.join(temp_label_dir, label_file), os.path.join(test_label_dir, label_file))

    # Удаляем временные каталоги
    shutil.rmtree(temp_image_dir)
    shutil.rmtree(temp_label_dir)

# Функция для упаковки датасета в ZIP-архив
def zip_dataset(dataset_dir, zip_file):
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dataset_dir)
                zipf.write(file_path, arcname)

# Создаем временные каталоги для сбора всех файлов
temp_image_dir = os.path.join(DATASET_DIR, "temp_images")
temp_label_dir = os.path.join(DATASET_DIR, "temp_labels")
os.makedirs(temp_image_dir, exist_ok=True)
os.makedirs(temp_label_dir, exist_ok=True)

# Собираем все файлы в временные каталоги
image_dirs = [
    os.path.join(DATASET_DIR, "train", "images"),
    os.path.join(DATASET_DIR, "valid", "images"),
    os.path.join(DATASET_DIR, "test", "images")
]

label_dirs = [
    os.path.join(DATASET_DIR, "train", "labels"),
    os.path.join(DATASET_DIR, "valid", "labels"),
    os.path.join(DATASET_DIR, "test", "labels")
]

for image_dir in image_dirs:
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                shutil.move(os.path.join(image_dir, filename), os.path.join(temp_image_dir, filename))

for label_dir in label_dirs:
    if os.path.exists(label_dir):
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                shutil.move(os.path.join(label_dir, filename), os.path.join(temp_label_dir, filename))

# Удаление дубликатов
if REMOVE_DUPLICATES:
    remove_duplicates(temp_image_dir, temp_label_dir)

# Удаление пустых меток
if REMOVE_EMPTY_LABELS:
    remove_empty_labels(temp_label_dir, temp_image_dir)

# Перераспределение файлов
if REDISTRIBUTE_FILES:
    redistribute_files(DATASET_DIR, temp_image_dir, temp_label_dir)

# Упаковка датасета в ZIP-архив
if ZIP_DATASET:
    zip_dataset(DATASET_DIR, "dataset.zip")

print("Процесс удаления дубликатов и пустых меток завершён.")
print("Файлы перераспределены в соответствии с заданными пропорциями.")
print(f"Датасет упакован в архив: dataset.zip")
