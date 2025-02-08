import pygame
import app
from spritesheet import Spritesheet

class Trainer:

    def __init__(self):
        self.sheet = Spritesheet('imgs/trainer_spritesheet.png')
        self.frames = [
            self.sheet.parse_sprite('trainer1.png'),
            self.sheet.parse_sprite('trainer2.png'),
            self.sheet.parse_sprite('trainer3.png'),
            self.sheet.parse_sprite('trainer4.png'),
            self.sheet.parse_sprite('trainer5.png')
        ]
        self.animation = [pygame.transform.scale(sprite, (300, 300)) for sprite in self.frames]
        self.showing = True
        self.throwing_ball = False
        self.x = 80
        self.frame_index = 0
        self.animation_speed = 200
        self.slide_speed = 20

def throw_ball():
    if app.trainer.throwing_ball:
        if app.current_time - app.last_update > app.trainer.animation_speed:
            app.trainer.frame_index += 1
            app.last_update = app.current_time

            if app.trainer.frame_index >= len(app.trainer.animation):
                app.trainer.frame_index = len(app.trainer.animation) - 1  

        if app.trainer.x > -300:
            app.trainer.x -= app.trainer.slide_speed
        else:
            app.trainer.showing = False
            app.trainer.throwing_ball = False

if app.trainer != None:
    app.canvas.blit(app.trainer.animation[app.trainer.frame_index], (app.trainer.x, app.canvas_height - 300))