import os
import shutil
import logging
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Путь к новому объединенному датасету
combined_path = 'ComboDataset'

# Создание структуры нового каталога
os.makedirs(os.path.join(combined_path, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(combined_path, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(combined_path, 'valid/images'), exist_ok=True)
os.makedirs(os.path.join(combined_path, 'valid/labels'), exist_ok=True)
os.makedirs(os.path.join(combined_path, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(combined_path, 'test/labels'), exist_ok=True)

# Функция для копирования изображений и меток с изменением идентификаторов классов
def copy_and_update_labels(src_images_path, src_labels_path, dest_images_path, dest_labels_path, class_mapping):
    for file_name in os.listdir(src_images_path):
        shutil.copy(os.path.join(src_images_path, file_name), dest_images_path)

    for file_name in os.listdir(src_labels_path):
        src_label_path = os.path.join(src_labels_path, file_name)
        dest_label_path = os.path.join(dest_labels_path, file_name)

        with open(src_label_path, 'r') as src_file:
            lines = src_file.readlines()

        with open(dest_label_path, 'w') as dest_file:
            for line in lines:
                parts = line.split()
                class_id = int(parts[0])
                new_class_id = class_mapping.get(class_id, class_id)
                parts[0] = str(new_class_id)
                dest_file.write(' '.join(parts) + '\n')

# Функция для загрузки data.yaml и получения соответствия классов
def load_class_mapping(dataset_path, current_class_mapping):
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(data_yaml_path, 'r') as file:
        data_yaml = YAML().load(file)

    class_names = data_yaml['names']
    class_mapping = {}

    if isinstance(class_names, dict):
        for class_id, class_name in class_names.items():
            if class_name not in current_class_mapping:
                current_class_mapping[class_name] = len(current_class_mapping)
            class_mapping[int(class_id)] = current_class_mapping[class_name]
    elif isinstance(class_names, list):
        for class_id, class_name in enumerate(class_names):
            if class_name not in current_class_mapping:
                current_class_mapping[class_name] = len(current_class_mapping)
            class_mapping[class_id] = current_class_mapping[class_name]

    return class_mapping, current_class_mapping

# Инициализация соответствия классов
current_class_mapping = {}

# Автоматическое определение всех датасетов, кроме CombinedDataset
datasets = [d for d in os.listdir('.') if os.path.isdir(d) and d != combined_path]

# Обработка каждого датасета
for dataset in datasets:
    dataset_path = dataset
    logging.info(f'Processing dataset: {dataset}')
    class_mapping, current_class_mapping = load_class_mapping(dataset_path, current_class_mapping)

    for split in ['train', 'valid', 'test']:
        src_images_path = os.path.join(dataset_path, split, 'images')
        src_labels_path = os.path.join(dataset_path, split, 'labels')
        dest_images_path = os.path.join(combined_path, split, 'images')
        dest_labels_path = os.path.join(combined_path, split, 'labels')

        copy_and_update_labels(src_images_path, src_labels_path, dest_images_path, dest_labels_path, class_mapping)

# Сортируем классы по ID
sorted_class_names = {v: k for k, v in sorted(current_class_mapping.items(), key=lambda item: item[1])}

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

with open(os.path.join(combined_path, 'data.yaml'), 'w') as file:
    yaml.dump(dataset_yaml, file)

logging.info('Dataset combination process completed successfully')
