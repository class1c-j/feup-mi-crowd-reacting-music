import pygame
from pygame.locals import *
from PIL import Image
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import cv2
import statistics
from generation import generate_music

model = YOLO("yolov8n-face.pt")

source = "video.mp4"  # Change to 0 to use webcam
video_capture = cv2.VideoCapture(source)
_, frame = video_capture.read()
height, width, _ = frame.shape

pygame.init()
window_width = 1200
window_height = 800
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Emotion Detection")

font = pygame.font.Font(None, 48)

clock = pygame.time.Clock()

pygame.mixer.init()

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    frame = np.rot90(frame)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    results = model.predict(frame, classes=0)
    number_of_people = len(results[0].boxes)

    emotions = []
    for i in range(number_of_people):
        x1, y1, x2, y2 = results[0].boxes[i].xyxy[0].int().tolist()
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_array = np.array(cropped_image)
        objs = DeepFace.analyze(
            img_path=cropped_array, actions=["emotion"], enforce_detection=False
        )
        dominant_emotion = objs[0]["dominant_emotion"]
        emotions.append(dominant_emotion)

    main_emotion = "Unknown"
    quadrant = 0
    if emotions:
        main_emotion = statistics.mode(emotions)
        high_valence = main_emotion in ["happy", "surprise", "neutral"]
        high_energy = number_of_people > 3

        quadrant = (
            1
            if not high_valence and high_energy
            else 2
            if high_energy and high_valence
            else 3
            if not high_valence and not high_energy
            else 4
        )

    volume = 1.0 if quadrant in [1, 2] else 0.5
    pygame.mixer.music.set_volume(volume)

    if not pygame.mixer.music.get_busy() and quadrant > 0:
        generate_music(quadrant)
        pygame.mixer.music.load("current.mid")
        pygame.mixer.music.play()

    scale_factor = min(window_width / width, window_height / height)
    scaled_width = int(scale_factor * width)
    scaled_height = int(scale_factor * height)
    scaled_frame = cv2.resize(frame, (scaled_width, scaled_height))

    frame_x = (window_width - scaled_width) // 2
    frame_y = (window_height - scaled_height) // 2

    image_surface = pygame.surfarray.make_surface(
        cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
    )

    screen.fill((0, 0, 0))
    emotion_surface = font.render(f"Emotion: {main_emotion}", True, (255, 255, 255))
    quadrant_surface = font.render(f"Quadrant: {quadrant}", True, (255, 255, 255))
    people_surface = font.render(f"People: {number_of_people}", True, (255, 255, 255))

    screen.fill((0, 0, 0))

    screen.blit(image_surface, (0, 0))

    text_x = window_width - 280
    screen.blit(emotion_surface, (text_x, 20))
    screen.blit(quadrant_surface, (text_x, 60))
    screen.blit(people_surface, (text_x, 100))

    pygame.display.flip()
    clock.tick(30)


video_capture.release()
pygame.quit()
