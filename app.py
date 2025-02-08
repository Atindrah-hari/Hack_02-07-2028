import pygame
import pygame.freetype

pygame.init()
pygame.freetype.init()

# global variables
canvas_width, canvas_height = 900, 600
window = pygame.display.set_mode((canvas_width, canvas_height))
canvas = pygame.Surface((canvas_width, canvas_height))
pygame.display.set_caption("TartanMon")
pokefont = pygame.freetype.Font("PokemonGb-RAeo.ttf")
running = True

trainer = None
curr_pokemon = None
hit_player_animation_active = False
hit_player_start_time = 0
hit_player_duration = 400 
hit_player_speed = 70
hit_player_frame_index = 0
current_time = pygame.time.get_ticks()
last_update = pygame.time.get_ticks()
text_box_active  = False
text_box_start_time = 0
text_box_duration = 2000
text_box_x = -400
text_box_final_x = 0
text_box_speed = 25
text_box_sliding_out = False
opponent = None
hit_opponent_animation_active = False
hit_opponent_start_time = 0
hit_opponent_duration = 400 
hit_opponent_speed = 70
hit_opponent_frame_index = 0
last_hit_update = pygame.time.get_ticks()

# michael
start_bg = pygame.image.load("imgs/startScreen.png")
guide_bg = pygame.image.load("imgs/guideScreen.png")
select_bg = pygame.image.load("imgs/selectScreen.png")
battle_bg = pygame.image.load("imgs/battle_background.jpeg")
battle_bg = pygame.transform.scale(battle_bg, (canvas_width, canvas_height))
win_bg = pygame.image.load("imgs/winScreen.png")
lose_bg = pygame.image.load("imgs/loseScreen.png")
bg = start_bg
charImg = pygame.image.load("imgs/Charmander.png")
bulbImg = pygame.image.load("imgs/Bulbasaur.png")
squirtImg = pygame.image.load("imgs/Squirtle.png")
charDeadImg = pygame.image.load("imgs/faintedChar.png")
bulbDeadImg = pygame.image.load("imgs/faintedBulb.png")
squirtDeadImg = pygame.image.load("imgs/faintedSquirt.png")
arrowImg = pygame.image.load("imgs/arrow.png")
in_start = True
in_guide = False
in_select = False
in_battle = False
in_win = False
in_lose = False
pickChar = False
pickBulb = False
pickSquirt = False
charDead = False
bulbDead = False
squirtDead = False
opponent_dead = False
curr_pokemon_dead = False
turn = True

mouseX = 1
mouseY = 1
mousePressed = False

def reset():
    global running, trainer, curr_pokemon, current_time
    global text_box_active, text_box_start_time, text_box_duration, text_box_x
    global text_box_final_x, text_box_speed, text_box_sliding_out, last_update
    global opponent, hit_opponent_animation_active, hit_opponent_start_time
    global hit_opponent_duration, hit_opponent_speed, hit_opponent_frame_index
    global last_hit_update, hit_player_animation_active, hit_player_start_time
    global hit_player_duration, hit_player_speed, hit_player_frame_index
    global in_start, in_guide, in_select, in_battle, in_win, in_lose, bg
    global pickChar, pickBulb, pickSquirt, charDead, bulbDead, squirtDead
    global opponent_dead, curr_pokemon_dead, turn
    running = True
    trainer =  None
    curr_pokemon = None
    hit_player_animation_active = False
    hit_player_frame_index = 0
    current_time = pygame.time.get_ticks()
    last_update = pygame.time.get_ticks()
    text_box_active  = False
    text_box_sliding_out = False
    opponent = None
    hit_opponent_animation_active = False
    hit_opponent_frame_index = 0
    last_hit_update = pygame.time.get_ticks()
    in_start = True
    in_guide = False
    in_select = False
    in_battle = False
    in_win = False
    in_lose = False
    pickChar = False
    pickBulb = False
    pickSquirt = False
    charDead = False
    bulbDead = False
    squirtDead = False
    opponent_dead = False
    curr_pokemon_dead = False
    turn = True
