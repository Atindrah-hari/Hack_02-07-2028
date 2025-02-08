import pygame
import random
import app
from spritesheet import Spritesheet
from player_mon import Pokemon

opponent_health_box = pygame.transform.scale(pygame.image.load('imgs/health_box.png'), (450, 150))

# hit opponent animation info
app.hit_opponent_sheet = Spritesheet('imgs/rattata_spritesheet.png')
app.hit_opponent_frames = [
    app.hit_opponent_sheet.parse_sprite('rattata1.png'),
    app.hit_opponent_sheet.parse_sprite('rattata2.png'),
    app.hit_opponent_sheet.parse_sprite('rattata3.png'),
    app.hit_opponent_sheet.parse_sprite('rattata4.png'),
]
app.hit_opponent_animation = [pygame.transform.scale(sprite, (300, 300)) for sprite in app.hit_opponent_frames]

def spawn_opponent():
    opponent_moves = {0: ['Tackle', 5], 1: ['Quick Attack', 5], 2: ['Bite', 5], 3: ['Take Down', 5]}
    app.opponent = Pokemon('rattata', 'imgs/rattata.png', 160, opponent_moves, 40)

def use_opponent_move():
    if app.opponent and app.curr_pokemon:
        move_index = random.randint(0, 3)
        app.opponent.curr_move, app.opponent.move_power = app.opponent.moves[move_index]
        app.text_box_active = True
        app.text_box_sliding_out = False
        app.text_box_x = -400
        app.text_box_start_time = app.current_time
        app.text = [f'{app.opponent.name} used', app.opponent.curr_move] 
        app.hit_player_animation_active = True
        app.hit_player_start_time = app.current_time
        app.hit_player_frame_index = 0
        app.last_player_hit_update = app.current_time

def hit_opponent():
    if app.opponent and app.curr_pokemon.curr_move and app.curr_pokemon.move_power:
        app.opponent.hp -= app.curr_pokemon.move_power
        if app.opponent.hp <= 0:
            app.opponent.hp = 0
            app.opponent.fainted = True
            app.text_box_active = True
            app.text = [f'{app.opponent.name} fainted']
        app.hit_opponent_animation_active = True
        app.hit_opponent_start_time = app.current_time
        app.hit_opponent_frame_index = 0
        app.last_hit_update = app.current_time
        app.curr_pokemon.curr_move = None
        app.curr_pokemon.move_power = None

def hit_opponent_sprite():
    if app.current_time - app.hit_opponent_start_time < app.hit_opponent_duration:
        if app.current_time - app.last_hit_update > app.hit_opponent_speed:
            app.hit_opponent_frame_index = (app.hit_opponent_frame_index + 1) % len(app.hit_opponent_animation)
            app.last_hit_update = app.current_time 
    else:
        app.hit_opponent_animation_active = False 

def opponent_fade_in():
    app.opponent.alpha += app.opponent.fade_speed
    app.opponent.alpha = min(app.opponent.alpha, 255)
    app.opponent.img.set_alpha(app.opponent.alpha)

def opponent_fade_out():
    app.opponent.alpha -= app.opponent.fade_speed
    app.opponent.alpha = max(0, app.opponent.alpha)
    app.opponent.img.set_alpha(app.opponent.alpha)
    app.canvas.blit(app.opponent.img, (520, 160))

def draw_opponent_mon():
    app.opponent.img.set_alpha(app.opponent.alpha)
    if app.hit_opponent_animation_active:
        hit_opponent_sprite()
        app.canvas.blit(app.hit_opponent_animation[app.hit_opponent_frame_index], (520, 160))
    elif app.opponent.fainted:
        if app.opponent.alpha > 0:
            opponent_fade_out()
            app.canvas.blit(app.opponent.img, (520, 160))
        else:
            app.opponent = None
            app.opponent_dead = True
    else:
        app.canvas.blit(app.opponent.img, (520, 160))

    if app.opponent and not app.opponent.fainted:
        if app.opponent.alpha < 255:
            opponent_fade_in()
        elif not app.opponent.fainted:
            draw_opponent_health_box()

def draw_opponent_health_box():
    # draw health box
    app.canvas.blit(opponent_health_box, (40, 120))
    # Pokemon name
    opponent_text, opponent_rect = app.pokefont.render(app.opponent.name.upper(), (0, 0, 0), size = 26)
    opponent_rect.topleft = (70, 150)
    app.canvas.blit(opponent_text, opponent_rect)
    # Pokemon health
    hp_text, hp_rect = app.pokefont.render('40', (0, 0, 0), size = 32)
    opponent_hp_text, opponent_hp_rect = app.pokefont.render(str(app.opponent.hp), (0, 0, 0), size = 32)
    hp_rect.topleft = (400, 230)
    opponent_hp_rect.center = (338, 242)
    app.canvas.blit(hp_text, hp_rect)
    app.canvas.blit(opponent_hp_text, opponent_hp_rect)
    # health bar
    if app.opponent.hp > 0:
        pygame.draw.line(app.canvas, (77, 205, 59), (230, 207), (230 + app.opponent.hp * 5.86, 207), 13)