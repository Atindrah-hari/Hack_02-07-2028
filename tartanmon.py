import pygame
import app
from screens import *
from trainer import *
from player_mon import *
from opponent_mon import *
from text_box import *

last_update = pygame.time.get_ticks() 
clock = pygame.time.Clock()
app.reset()

while app.running:

    app.canvas.blit(app.bg, (0,0))

    right_hand = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False 
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if app.in_start and pressedStart(event.pos):
                toGuide()
            elif (app.in_win or app.in_lose) and pressedPlayAgain(event.pos):
                toStart()
                app.reset()
            elif app.in_guide and pressedGotIt(event.pos):
                toSelect()
            elif app.in_select:
                if pressedBack(event.pos):
                    toStart()
                if hoverBulb(event.pos):
                    if not(app.bulbDead):
                        toBattle()
                        if app.curr_pokemon_dead or app.curr_pokemon == None: 
                            app.trainer = Trainer()
                        app.pickChar = False
                        app.pickBulb = True
                        app.pickSquirt = False
                        app.curr_pokemon_dead = False
                elif hoverChar(event.pos): 
                    if not(app.charDead):
                        toBattle()
                        if app.curr_pokemon_dead or app.curr_pokemon == None: 
                            app.trainer = Trainer()
                        app.pickChar = True
                        app.pickBulb = False
                        app.pickSquirt = False
                        app.curr_pokemon_dead = False
                elif hoverSquirt(event.pos): 
                    if not(app.squirtDead):
                        toBattle()
                        if app.curr_pokemon_dead or app.curr_pokemon == None: 
                            app.trainer = Trainer()
                        app.pickChar = False
                        app.pickBulb = False
                        app.pickSquirt = True
                        app.curr_pokemon_dead = False
        elif event.type == pygame.KEYDOWN: #temporary for fainting pokemon and going to end screen
            if event.key == pygame.K_r:
                app.reset()
                toStart()
            #-----------------------------------------------------------------------------------
            if app.in_battle:
                if event.key == pygame.K_BACKSPACE:
                    toSelect()
                if app.curr_pokemon == None: # spawn in opponent and player
                    if event.key == pygame.K_SPACE:
                        app.trainer.throwing_ball = True
                        throw_ball()
                        spawn_player_mon()
                        if app.opponent == None: spawn_opponent()
                else:
                    # player attacks
                    if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4) and app.opponent:
                        use_player_move(event)
                    # opponent attacks
                    elif event.key == pygame.K_h and app.curr_pokemon.hp > 0 and app.opponent != None:
                        use_opponent_move()
                    if app.hit_opponent_animation_active and app.opponent:
                        hit_opponent()
                    if app.hit_player_animation_active:
                        hit_player()

    if app.in_battle:
        app.current_time = pygame.time.get_ticks()
        
        if app.trainer.showing:
            throw_ball()
            app.canvas.blit(app.trainer.animation[app.trainer.frame_index], (app.trainer.x, app.canvas_height - 300))

        if app.curr_pokemon is not None: draw_player_mon()

        if app.opponent is not None: draw_opponent_mon()

        if app.text_box_active: draw_text_box()

        if app.charDead and app.squirtDead and app.bulbDead: toLose()
        elif app.opponent_dead: toWin()
        elif app.curr_pokemon_dead: toSelect()

    if app.in_select:
        mousePos = pygame.mouse.get_pos()
        instr_text, instr_rect = app.pokefont.render("CLICK TO CHOOSE", (0,0,0), size = 18)
        instr_rect.center = (450, 535)
        app.canvas.blit(instr_text, instr_rect)

        if hoverBulb(mousePos):
            app.canvas.blit(app.arrowImg, (140, 180))
            if app.bulbDead:
                warn_text, warn_rect = app.pokefont.render("FAINTED! CAN'T CHOOSE", (0,0,0), size = 18)
                warn_rect.center = (450, 565)
                app.canvas.blit(warn_text, warn_rect)
        elif hoverChar(mousePos):
            app.canvas.blit(app.arrowImg, (405, 180))
            if app.charDead:
                warn_text, warn_rect = app.pokefont.render("FAINTED! CAN'T CHOOSE", (0,0,0), size = 18)
                warn_rect.center = (450, 565)
                app.canvas.blit(warn_text, warn_rect)
        elif hoverSquirt(mousePos):
            app.canvas.blit(app.arrowImg, (680, 180))
            if app.squirtDead:
                warn_text, warn_rect = app.pokefont.render("FAINTED! CAN'T CHOOSE", (0,0,0), size = 18)
                warn_rect.center = (450, 565)
                app.canvas.blit(warn_text, warn_rect)
                
        else:
            app.canvas.fill((0, 0, 0))
            app.canvas.blit(app.bg, (0,0))

        if app.bulbDead:
            app.canvas.blit(app.bulbDeadImg, (88, 245))
        else:
            app.canvas.blit(app.bulbImg, (88, 245))
        
        if app.charDead:
            app.canvas.blit(app.charDeadImg, (375, 240))
        else: 
            app.canvas.blit(app.charImg, (375, 240))
        
        if app.squirtDead:
            app.canvas.blit(app.squirtDeadImg, (625, 240))
        else:
            app.canvas.blit(app.squirtImg, (625, 240))

    if app.opponent_dead: pass

    app.window.blit(app.canvas, (0,0))
    pygame.display.update()
    clock.tick(60)

pygame.quit()