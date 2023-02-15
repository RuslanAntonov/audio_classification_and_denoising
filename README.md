# Классификация шума на mel-спектрограмме и его удаление

Решение состоит из двух частей: 
1. Классификация шума
2. Denoising

Папки train и val должны располагаться в корневой папке проекта.

### Классификация
Задание предполагает решение задачи бинарной классификации, необходимо определить, является ли запись зашумленной или нет. 

Для подготовки данных необходимо запустить class_prepair_data.py. Скрипт загружает mel-диограммы в один массив, разбивает их на массивы фиксированной длины (250*80), заполняет их нулями в случае если их длина меньше 250 и генерирует метки класса.

Для тренировки необходимо запустить class_train.py.

Для тестирования необходимо запустить class_pred.py.

### Denoising
Алгоритм считывает зашумленную mel-спектрограмму и генерирует очищенную версию.
Для подготовки данных необходимо запустить denoise_prepair_data.py. Алгоритм генерирует два набора данных, содержащих зашумленные и чистые данные.
Запуск скриптов для тренировки и тестирования аналогичен классификации.
