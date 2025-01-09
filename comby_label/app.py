import os
import logging
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Путь к датасету
dataset_path = 'container'

# Функция для обновления меток
def update_labels(labels_path):
    for file_name in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, file_name)

        with open(label_file_path, 'r') as file:
            lines = file.readlines()

        with open(label_file_path, 'w') as file:
            for line in lines:
                parts = line.split()
                class_id = int(parts[0])
                # Объединяем классы 0 и 1 в один класс с идентификатором 0
                if class_id in [0, 1]:
                    parts[0] = '0'
                file.write(' '.join(parts) + '\n')

# Обновление меток для всех разделов датасета
for split in ['train', 'valid', 'test']:
    labels_path = os.path.join(dataset_path, split, 'labels')
    if os.path.exists(labels_path):
        logging.info(f'Updating labels in {split} split')
        update_labels(labels_path)
    else:
        logging.warning(f'Labels path does not exist: {labels_path}')

# Обновление data.yaml
data_yaml_path = os.path.join(dataset_path, 'data.yaml')
if os.path.exists(data_yaml_path):
    with open(data_yaml_path, 'r') as file:
        data_yaml = YAML().load(file)

    # Обновляем блок names
    data_yaml['names'] = {0: 'can'}

    # Используем ruamel.yaml для сохранения порядка ключей
    yaml = YAML()
    yaml.default_flow_style = False

    with open(data_yaml_path, 'w') as file:
        yaml.dump(data_yaml, file)

    logging.info('data.yaml updated successfully')
else:
    logging.warning(f'data.yaml does not exist: {data_yaml_path}')

logging.info('Label merging process completed successfully')
