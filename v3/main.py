from pywebio.output import *
from pywebio.input import *
import pywebio
from ultralytics import YOLO
import cv2
import numpy as np
import os
from time import sleep
from gtts import gTTS
from playsound3 import playsound



pywebio.config(theme='dark')

try:
    hello = gTTS('Добрый день повелитель', lang='ru')
    hello.save('hello.mp3')
except:
    put_error('Не известная ошибка проверьте интернет!!!')
    print('[Error] Не известная ошибка проверьте интернет!!!')

def video_detection(file,speed=True ,YOLO_model='yolov8n.pt'):
    if speed == True:
        print('[Log] Пожалуйста подождите!!!')
        put_code('Пожалуйста подождите!!!')
        from ultralytics import YOLO
        import cv2
        import numpy as np

        # Загрузка модели YOLOv8
        model = YOLO('yolov8n.pt')

        # Список цветов для различных классов
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
            (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
            (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
        ]

        # Открытие исходного видеофайла
        input_video_path = file
        capture = cv2.VideoCapture(input_video_path)

        # Чтение параметров видео
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Настройка выходного файла
        name = file.split('.')[0]
        output_video_path = f'{name}_detect.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while True:
            # Захват кадра
            ret, frame = capture.read()
            if not ret:
                break

            # Обработка кадра с помощью модели YOLO
            results = model(frame)[0]

            # Получение данных об объектах
            classes_names = results.names
            classes = results.boxes.cls.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
            # Рисование рамок и подписей на кадре
            for class_id, box, conf in zip(classes, boxes, results.boxes.conf):
                if conf>0.5:
                    class_name = classes_names[int(class_id)]
                    color = colors[int(class_id) % len(colors)]
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Запись обработанного кадра в выходной файл
            writer.write(frame)

        # Освобождение ресурсов и закрытие окон
        capture.release()
        writer.release()
    elif speed == False:
        from ultralytics import YOLO
        import cv2
        import numpy as np

        model = YOLO('yolov8n.pt')

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
            (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
            (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
        ]

        capture = cv2.VideoCapture(file)


            # Чтение параметров видео
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Настройка выходного файла
        name = file.split('.')[0]
        output_video_path = f'{name}_detect.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while True:
            ret, frame = capture.read()
            results = model(frame)[0]

            classes_names = results.names
            classes = results.boxes.cls.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

            for class_id, box, conf in zip(classes, boxes, results.boxes.conf):
                if conf > 0.5:
                    class_name = classes_names[int(class_id)]
                    color = colors[int(class_id) % len(colors)]
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            writer.write(frame)
            cv2.imshow('Object detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

def not_cam_detection(file, YOLO_model='yolo11m-seg.pt'):
    # Загрузка модели YOLOv8
    model = YOLO('yolo11m-seg.pt')
    print()

    # Список цветов для различных классов
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
        (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
        (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
    ]


    # Функция для обработки изображения
    def process_image(image_path,YOLO_model='yolov8n.pt'):
        # Загрузка изображения
        image = cv2.imread(image_path)
        results = model(image)[0]

        # Получение оригинального изображения и результатов
        image = results.orig_img
        classes_names = results.names
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

        # Подготовка словаря для группировки результатов по классам
        grouped_objects = {}

        # Рисование рамок и группировка результатов
        for class_id, box in zip(classes, boxes):
            class_name = classes_names[int(class_id)]
            color = colors[int(class_id) % len(colors)]  # Выбор цвета для класса
            if class_name not in grouped_objects:
                grouped_objects[class_name] = []
                grouped_objects[class_name].append(box)

            # Рисование рамок на изображении
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Сохранение измененного изображения
            new_image_path = os.path.splitext(image_path)[0] + '_output' + os.path.splitext(image_path)[1]
            cv2.imwrite(new_image_path, image)

        put_text(f"Обработано {image_path}:")
        print(f"[Log] Обработано {image_path}:")
        put_text(f"Сохранено изображение с боксами в {new_image_path}")
        print(f"[Log] Сохранено изображение с боксами в {new_image_path}")
        put_text('Пожалуйста, подождите...')
        print('[Log] Пожалуйста, подождите...')
        sleep(3)
        put_text('Готово...')
        os.system(new_image_path)
    process_image(file)

def cam_detection(index=0, YOLO_model='yolov8n.pt'):
    from ultralytics import YOLO
    import cv2
    import numpy as np

    model = YOLO('yolov8n.pt')

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
        (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
        (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
    ]

    capture = cv2.VideoCapture(int(index))

    while True:
        ret, frame = capture.read()
        results = model(frame)[0]

        classes_names = results.names
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

        for class_id, box, conf in zip(classes, boxes, results.boxes.conf):
            if conf > 0.5:
                class_name = classes_names[int(class_id)]
                color = colors[int(class_id) % len(colors)]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('Object detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()




playsound('hello.mp3')
def main():
    put_code('1 - Изображение с камеры\n2 - Изображение с фото\n3 - Изображение с видео')

    mode = input('Введите режим: ')
    put_text(mode)
    if mode == '1':
        put_code('Введите индекс камеры!!!')
        cam_index = input('Изображение с камеры!!!')
        put_text(cam_index)
        cam_detection(index=cam_index)
        put_code('Введите индекс камеры!!!')
        os.system('cls')
        main()
    elif mode == '2':
        filename = input('Введите имя файла: ')
        print(not_cam_detection(YOLO_model='yolo11m-seg.pt',file=filename))
        os.system('cls')
        main()
    elif mode == '3':
        filename = input('Введите имя файла: ')
        speed = input('Включить скорость?')
        if speed == 'False':
            video_detection(speed=False, file=filename)
        elif speed == 'True':
            video_detection(speed=True, file=filename)
        name = filename.split('.')[0]
        put_code('Готово!!!')
        print('[Log] Готово!!!')
        sleep(1)
        playsound(f'{name}_detect.mp4')
        os.system('cls')
        main()
    elif mode == '$exit':
        put_success('Вы успешно вышли!!!')
        exit()


if __name__ == '__main__':
    main()