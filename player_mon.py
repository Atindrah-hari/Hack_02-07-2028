import pygame
import app
from spritesheet import Spritesheet

player_health_box = pygame.transform.scale(pygame.image.load('imgs/health_box.png'), (450, 150))

app.hit_bulbasaur_sheet = Spritesheet('imgs/bulbasaur_spritesheet.png')
app.hit_bulbasaur_frames = [
    app.hit_bulbasaur_sheet.parse_sprite('hit1.png'),
    app.hit_bulbasaur_sheet.parse_sprite('hit2.png'),
    app.hit_bulbasaur_sheet.parse_sprite('hit3.png'),
    app.hit_bulbasaur_sheet.parse_sprite('hit4.png'),
]
app.hit_charmander_sheet = Spritesheet('imgs/charmander_spritesheet.png')
app.hit_charmander_frames = [
    app.hit_charmander_sheet.parse_sprite('hit1.png'),
    app.hit_charmander_sheet.parse_sprite('hit2.png'),
    app.hit_charmander_sheet.parse_sprite('hit3.png'),
    app.hit_charmander_sheet.parse_sprite('hit4.png'),
]
app.hit_squirtle_sheet = Spritesheet('imgs/squirtle_spritesheet.png')
app.hit_squirtle_frames = [
    app.hit_squirtle_sheet.parse_sprite('hit1.png'),
    app.hit_squirtle_sheet.parse_sprite('hit2.png'),
    app.hit_squirtle_sheet.parse_sprite('hit3.png'),
    app.hit_squirtle_sheet.parse_sprite('hit4.png'),
]

class Pokemon:

    def __init__(self, name, img_url, y, moves, hp):
        self.name = name
        self.img = pygame.transform.scale(pygame.image.load(img_url), (300, 300))
        self.y = y
        self.alpha = 0
        self.fade_speed = 10
        self.hp = hp
        self.fainted = False
        self.moves = moves
        self.curr_move = None
        self.move_power = None
    
    def swords_dance(self):
        for move_name in self.moves:
            if move_name != 3: self.moves[move_name][1] += 2

def spawn_player_mon():
    if app.pickBulb:
        moves = {0: ['Tackle', 2], 1: ['Vine Whip', 5], 2: ['Razor Leaf', 7], 3: ['Swords Dance', 0]}
        app.curr_pokemon = Pokemon('bulbasaur', 'imgs/bulbasaur_back.png', 380, moves, 20)
        app.hit_player_animation = [pygame.transform.scale(sprite, (300, 300)) for sprite in app.hit_bulbasaur_frames]
    elif app.pickChar:
        moves = {0: ['Scratch', 2], 1: ['Ember', 5], 2: ['Flamethrower', 7], 3: ['Swords Dance', 0]}
        app.curr_pokemon = Pokemon('charmander', 'imgs/charmander_back.png', 350, moves, 20)
        app.hit_player_animation = [pygame.transform.scale(sprite, (300, 300)) for sprite in app.hit_charmander_frames]
    elif app.pickSquirt:
        moves = {0: ['Tail Whip', 2], 1: ['Water Gun', 5], 2: ['Hydro Pump', 7], 3: ['Swords Dance', 0]}
        app.curr_pokemon = Pokemon('squirtle', 'imgs/squirtle_back.png', 380, moves, 20)
        app.hit_player_animation = [pygame.transform.scale(sprite, (300, 300)) for sprite in app.hit_squirtle_frames]

def use_player_move(current_gesture):
    if current_gesture == 'fist_palm': key_index = 0
    elif current_gesture == 'Triangle': key_index = 1
    elif current_gesture == 'dragonBall': key_index = 2
    elif current_gesture == 'Fly': key_index = 3
    if app.curr_pokemon and key_index in app.curr_pokemon.moves:
        app.curr_pokemon.curr_move, app.curr_pokemon.move_power = app.curr_pokemon.moves[key_index]
        app.text_box_active = True
        app.text_box_sliding_out = False
        app.text_box_x = -400
        app.text_box_start_time = app.current_time
        app.text = [f'{app.curr_pokemon.name} used', app.curr_pokemon.curr_move] 
        if app.curr_pokemon.curr_move == 'Swords Dance':
            app.curr_pokemon.swords_dance()
        else:
            print(app.curr_pokemon.curr_move)
            app.hit_opponent_animation_active = True
            app.hit_opponent_start_time = app.current_time
            app.hit_opponent_frame_index = 0
            app.last_hit_update = app.current_time

def hit_player():
    if app.opponent and app.opponent.curr_move and app.opponent.move_power:
        app.curr_pokemon.hp -= app.opponent.move_power
        if app.curr_pokemon.hp <= 0:
            app.curr_pokemon.hp = 0
            app.curr_pokemon.fainted = True
            app.text_box_active = True
            app.text = [f'{app.curr_pokemon.name} fainted']
        app.hit_player_animation_active = True
        app.hit_player_start_time = app.current_time
        app.hit_player_frame_index = 0
        app.last_player_hit_update = app.current_time
        app.opponent.curr_move = None
        app.opponent.move_power = None

def hit_player_sprite(): 
    if app.current_time - app.hit_player_start_time < app.hit_player_duration:
        if app.current_time - app.last_player_hit_update > app.hit_player_speed:
            app.hit_player_frame_index = (app.hit_player_frame_index + 1) % len(app.hit_player_animation)
            app.last_player_hit_update = app.current_time
    else:
        app.hit_player_animation_active = False

def player_fade_in():
    app.curr_pokemon.alpha += app.curr_pokemon.fade_speed
    app.curr_pokemon.alpha = min(app.curr_pokemon.alpha, 255)
    app.curr_pokemon.img.set_alpha(app.curr_pokemon.alpha)

def player_fade_out():
    app.curr_pokemon.alpha -= app.curr_pokemon.fade_speed
    app.curr_pokemon.alpha = max(0, app.curr_pokemon.alpha)
    app.curr_pokemon.img.set_alpha(app.curr_pokemon.alpha)
    app.canvas.blit(app.curr_pokemon.img, (80, app.curr_pokemon.y))

def draw_player_mon():
    app.curr_pokemon.img.set_alpha(app.curr_pokemon.alpha)
    if app.hit_player_animation_active:
        hit_player_sprite()
        app.canvas.blit(app.hit_player_animation[app.hit_player_frame_index], (80, app.curr_pokemon.y))
    elif app.curr_pokemon.fainted:
        if app.curr_pokemon.alpha > 0:
            player_fade_out()
            app.canvas.blit(app.curr_pokemon.img, (80, app.curr_pokemon.y))
        else:
            app.curr_pokemon = None
            app.curr_pokemon_dead = True
            if app.pickChar: app.charDead = True
            elif app.pickBulb: app.bulbDead = True
            elif app.pickSquirt: app.squirtDead = True
    else:
        app.canvas.blit(app.curr_pokemon.img, (80, app.curr_pokemon.y))

    if app.curr_pokemon and not app.curr_pokemon.fainted:
        if app.curr_pokemon.alpha < 255:
            player_fade_in()
        elif not app.curr_pokemon.fainted:
            draw_player_health_box()

def draw_player_health_box():
    # draw health box
    app.canvas.blit(player_health_box, (430, 420))
    # Pokemon name
    name_text, name_rect = app.pokefont.render(app.curr_pokemon.name.upper(), (0, 0, 0), size = 26)
    name_rect.topleft = (460, 450)
    app.canvas.blit(name_text, name_rect)
    # Pokemon health
    hp_text, hp_rect = app.pokefont.render('20', (0, 0, 0), size = 32)
    curr_hp_text, curr_hp_rect = app.pokefont.render(str(app.curr_pokemon.hp), (0, 0, 0), size = 32)
    hp_rect.topleft = (790, 530)
    curr_hp_rect.center = (728, 542)
    app.canvas.blit(hp_text, hp_rect)
    app.canvas.blit(curr_hp_text, curr_hp_rect)
    # health bar
    if app.curr_pokemon.hp > 0:
        pygame.draw.line(app.canvas, (77, 205, 59), (620, 507), (620 + app.curr_pokemon.hp * 11.8, 507), 13)