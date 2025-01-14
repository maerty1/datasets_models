Инструменты для датасета

Обработка и Аугментация Датасетов

Этот репозиторий содержит скрипты для обработки и аугментации датасетов для задач детекции объектов. Скрипты выполняют различные задачи, такие как детекция объектов на изображениях, применение аугментаций, удаление дубликатов, обработка пустых меток, перераспределение файлов и объединение нескольких датасетов.

Возможности

Детекция Объектов: Использует модель YOLO для детекции объектов на изображениях.
Аугментация: Применяет различные аугментации к изображениям с использованием библиотеки Albumentations.
Очистка Датасета: Удаляет дублирующиеся изображения и пустые метки.
Перераспределение Файлов: Перераспределяет изображения и метки в обучающую, валидационную и тестовую выборки.
Объединение Датасетов: Объединяет несколько датасетов в один с согласованными метками классов.
Скрипты

1. Train - Детекция Объектов и Аугментация

Этот скрипт обрабатывает изображения из исходного каталога, детектирует объекты с использованием модели YOLO, применяет аугментации и сохраняет результаты в структурированном каталоге датасета.

Ключевые Возможности:

Модель YOLO: Загружает предобученную модель YOLO для детекции объектов.
Аугментации: Применяет различные аугментации, такие как горизонтальное и вертикальное отражение, поворот, изменение яркости/контраста и т.д.
Структура Датасета: Организует изображения и метки в обучающую, валидационную и тестовую выборки.
Конфигурация YAML: Генерирует файл data.yaml с именами классов и путями к датасету.
<br>
2. Delete_dub - Очистка и Перераспределение Датасета

Этот скрипт очищает датасет, удаляя дублирующиеся изображения и пустые метки, перераспределяет файлы в обучающую, валидационную и тестовую выборки и опционально упаковывает датасет в ZIP-архив.

Ключевые Возможности:

Удаление Дубликатов: Использует предобученную модель ResNet для извлечения признаков и удаления дублирующихся изображений.
Удаление Пустых Меток: Удаляет изображения с пустыми файлами меток.
Перераспределение Файлов: Перераспределяет изображения и метки в обучающую, валидационную и тестовую выборки.
ZIP-архив: Опционально упаковывает датасет в ZIP-архив.
<br>
3. CombinedDataset - Объединение Датасетов

Этот скрипт объединяет несколько датасетов в один с согласованными метками классов. Он обновляет идентификаторы классов в файлах меток, чтобы обеспечить согласованность в объединенном датасете.

Ключевые Возможности:

Соответствие Классов: Загружает имена классов из файла data.yaml каждого датасета и сопоставляет их с согласованными идентификаторами.
Структура Датасета: Копирует изображения и метки из каждого датасета в каталог объединенного датасета.
Конфигурация YAML: Генерирует файл data.yaml для объединенного датасета с согласованными именами классов.
<br>
4. Web - Просмотр Датасета

Этот раздел позволяет просматривать датасет через веб-интерфейс.

<br>
5. Обновление меток
Проходит по всем файлам меток в папках train, valid и test.
Читает каждый файл меток, обновляет идентификаторы классов, объединяя классы 0 и 1 в один класс с идентификатором 0.
Записывает обновленные метки обратно в файлы.

Обновление data.yaml:

Загружает файл data.yaml и обновляет блок names, устанавливая имя класса с идентификатором 0 как 'can'.
Сохраняет обновленный файл data.yaml, сохраняя порядок ключей.


