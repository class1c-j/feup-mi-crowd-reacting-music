import pygame
pygame.init()

pygame.mixer.music.load("output.mid")
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.wait(1000)