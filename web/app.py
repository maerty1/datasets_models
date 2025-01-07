from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import cv2
import numpy as np
import yaml

app = Flask(__name__)

DATASET_DIR = "dataset"
YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

# Загрузка названий классов из файла dataset.yaml
with open(YAML_PATH, 'r') as file:
    dataset_yaml = yaml.safe_load(file)
    class_names = dataset_yaml['names']

@app.route('/')
def index():
    image_files = []
    for split in ['train', 'valid', 'test']:
        images_path = os.path.join(DATASET_DIR, split, 'images')
        for filename in os.listdir(images_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(split, 'images', filename))

    # Ограничиваем количество отображаемых изображений до 100
    image_files = image_files[:100]

    return render_template('index.html', image_files=image_files, class_names=class_names)

@app.route('/image/<path:filename>')
def image(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/annotated_image/<path:filename>')
def annotated_image(filename):
    image_path = os.path.join(DATASET_DIR, filename)
    label_path = os.path.join(DATASET_DIR, filename.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

    # Проверка существования файла метки
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return send_from_directory(DATASET_DIR, filename)

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Invalid label format: {line}")
                continue

            cls = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

            # Преобразование нормализованных координат в абсолютные
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            x1 = int(x_center - bbox_width / 2)
            y1 = int(y_center - bbox_height / 2)
            x2 = int(x_center + bbox_width / 2)
            y2 = int(y_center + bbox_height / 2)

            # Проверка координат
            if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
                print(f"Invalid coordinates: {x1}, {y1}, {x2}, {y2}")
                continue

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_names[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
