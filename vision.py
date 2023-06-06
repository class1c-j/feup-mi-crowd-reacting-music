import pygame
from pygame.locals import *
from PIL import Image
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import cv2
import statistics
from generation import generate_music

model = YOLO("yolov8n-face.pt")  # load an official model

# OpenCV webcam capture
video_capture = cv2.VideoCapture(0)
_, frame = video_capture.read()
height, width, _ = frame.shape

pygame.init()
screen_width = width + 300  # Width of the pygame screen
screen_height = height
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Emotion Detection")

font = pygame.font.Font(None, 36)  # Create a font object for text rendering

clock = pygame.time.Clock()

pygame.mixer.init()

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame = np.rot90(frame)  # Rotate the frame 90 degrees

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

    if not pygame.mixer.music.get_busy() and quadrant > 0:
        generate_music(quadrant)
        pygame.mixer.music.load("current.mid")
        pygame.mixer.music.play()

    # Create surfaces for image, text, and button
    image_surface = pygame.surfarray.make_surface(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    emotion_surface = font.render(f"Emotion: {main_emotion}", True, (255, 255, 255))
    quadrant_surface = font.render(f"Quadrant: {quadrant}", True, (255, 255, 255))
    people_surface = font.render(f"People: {number_of_people}", True, (255, 255, 255))

    screen.fill((0, 0, 0))  # Clear the screen

    # Blit image surface on the left side
    screen.blit(image_surface, (0, 0))

    # Blit text surface and button surface on the right side
    screen.blit(emotion_surface, (width + 20, 20))
    screen.blit(quadrant_surface, (width + 20, 50))
    screen.blit(people_surface, (width + 20, 80))

    pygame.display.flip()
    clock.tick(30)


video_capture.release()
pygame.quit()
