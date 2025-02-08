import pygame
import app

def draw_text_box():

    if app.current_time - app.text_box_start_time > app.text_box_duration:
        if not app.text_box_sliding_out:
            app.turn = not app.turn
            app.text_box_sliding_out = True

    if app.text_box_sliding_out:
        if app.text_box_x > -400:
            app.text_box_x -= app.text_box_speed
        else:
            app.text_box_active = False
            app.text_box_sliding_out = False
    else:
        if app.text_box_x < app.text_box_final_x:
            app.text_box_x += app.text_box_speed
        else:
            app.text_box_x = app.text_box_final_x

    pygame.draw.rect(app.canvas, (0, 0, 0), (app.text_box_x, 0, 407, 107))
    pygame.draw.rect(app.canvas, (250, 250, 250), (app.text_box_x + 3, 3, 400, 100))

    for i, line in enumerate(app.text):
        text, rect = app.pokefont.render(line.upper(), (0, 0, 0), size=20)
        rect.topleft = (app.text_box_x + 30, 37 * i + 20)
        app.canvas.blit(text, rect)