import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import tkinter as tk

# Инициализация MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

def update_camera_frame(canvas, cap, root, describe_pose):
    ret, frame = cap.read()
    if ret:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            describe_pose(results.pose_landmarks)

        # Конвертация изображения для Tkinter
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tk = ImageTk.PhotoImage(image_pil)

        # Обновление изображения на холсте Tkinter
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas.image = image_tk

        # Запланировать следующий вызов update_camera_frame
    root.after(10, lambda: update_camera_frame(canvas, cap, root, describe_pose))

def start_camera(canvas, root, describe_pose):
    cap = cv2.VideoCapture(0)
    update_camera_frame(canvas, cap, root, describe_pose)
    return cap

