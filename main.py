
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import image_processing
import camera_processing
import os
import sys
import subprocess


def select_image(canvas):
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("BMP", "*.bmp"),
            ("GIF", "*.gif"),
            ("Все файлы", "*.*"),
        ]
    )
    if not file_path:
        return

    image_bgr, results = image_processing.process_image(file_path)
    if not results:
        return

    # Аннотированное изображение
    annotated_image = image_processing.draw_landmarks_on_image(image_bgr.copy(), results.pose_landmarks)
    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Получаем размеры canvas и изображения
    canvas.update()  # чтобы гарантировать получение актуальных размеров
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    img_width = image_tk.width()
    img_height = image_tk.height()
    x = (canvas_width - img_width) // 2
    y = (canvas_height - img_height) // 2

    canvas.create_image(x, y, anchor=tk.NW, image=image_tk)
    canvas.image = image_tk

    # Проверка видимости лица И сохранение отчёта
    image_height, image_width, _ = image_bgr.shape
    if image_processing.is_face_fully_visible(results.pose_landmarks, image_width, image_height):
        image_processing.save_results_to_doc(annotated_image, "results.docx", results.pose_landmarks)
    else:
        print("Лицо не полностью видно.")

def open_report():
    report_path = "results.docx"
    if os.path.exists(report_path):
        if sys.platform.startswith("darwin"):  # macOS
            subprocess.call(["open", report_path])
        elif sys.platform.startswith("linux"):
            subprocess.call(["xdg-open", report_path])
        elif sys.platform.startswith("win"):
            os.startfile(report_path)
        else:
            messagebox.showinfo("Ошибка", "Открытие файлов не поддерживается на этой ОС.")
    else:
        messagebox.showerror("Ошибка", f"Файл отчета {report_path} не найден.")

def show_info():
    # Создаем отдельное окно
    info_win = tk.Toplevel()
    info_win.title("Информация")
    info_win.geometry("700x500")

    # Читаем текст из файла
    try:
        with open('info.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception:
        text = 'info.txt не найден.'

    # Cоздаём фрейм для скроллируемого текста
    frame = tk.Frame(info_win, bg="#EEEEEE")
    frame.pack(fill=tk.BOTH, expand=1)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text_widget = tk.Text(
        frame,
        font=('Arial', 16),
        wrap='word',
        yscrollcommand=scrollbar.set,
        bg='#EEEEEE',
        relief='flat',
        bd=0
    )
    text_widget.insert('1.0', text)
    text_widget.configure(state='disabled')
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, padx=10, pady=10)
    scrollbar.config(command=text_widget.yview)

def capture_and_detect_face():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Ошибка", "Не удалось захватить изображение с камеры.")
        return

    # Сохранение изображения
    save_path = "captured_image.png"
    cv2.imwrite(save_path, frame)
    if os.path.exists(save_path):
        print(f"Изображение сохранено в {save_path}")

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = image_processing.pose.process(image_rgb)

    if not results.pose_landmarks:
        messagebox.showinfo("Результаты", "Человек не найден. Пожалуйста, встаньте перед камерой.")
        return

    image_height, image_width, _ = frame.shape

    if image_processing.is_face_fully_visible(results.pose_landmarks, image_width, image_height):
        pass
        #messagebox.showinfo("Результаты", "Лицо полностью видно.")
    else:
        # Обновленная логика подсказок
        nose_landmark = results.pose_landmarks.landmark[image_processing.mp_pose.PoseLandmark.NOSE]
        move_message = "Пожалуйста, двигайтесь: "

        if nose_landmark.x < 0.3:
            move_message += "правее "
        elif nose_landmark.x > 0.7:
            move_message += "левее "

        if nose_landmark.y < 0.3:
            move_message += "ниже "
        elif nose_landmark.y > 0.7:
            move_message += "выше "

        messagebox.showinfo("Результаты",
                            move_message.strip() or "Лицо частично видно, попробуйте немного подвигаться.")

root = tk.Tk()
root.title('Приложение для распознавания частей тела')

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

btn_select_image = tk.Button(root, text="Выбрать изображение", command=lambda: select_image(canvas))
btn_select_image.pack(side=tk.LEFT, padx=10)

#btn_find_face = tk.Button(root, text="Найти лицо", command=capture_and_detect_face)
#btn_find_face.pack(side=tk.LEFT, padx=10)

#btn_open_camera = tk.Button(root, text="Открыть камеру", command=lambda: camera_processing.start_camera(canvas, root, image_processing.describe_pose))
#btn_open_camera.pack(side=tk.RIGHT, padx=10)

btn_open_report = tk.Button(root, text="Открыть отчет", command=open_report)
btn_open_report.pack(side=tk.LEFT, padx=10)

#btn_open_info = tk.Button(root, text="Информация", command=show_info)
#btn_open_info.pack(side=tk.LEFT, padx=10)

root.mainloop()
