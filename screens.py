import app

def toStart():
    app.in_start = True
    app.in_guide = False
    app.in_select = False
    app.in_battle = False
    app.in_win = False
    app.in_lose = False
    app.canvas.fill((0, 0, 0))
    app.bg = app.start_bg

def pressedStart(mousePos):
    mouseX, mouseY = mousePos
    if (353 <= mouseX <= 548 and 438 <= mouseY <= 515):
        return True
    return False

def toGuide():
    app.in_start = False
    app.in_guide = True
    app.in_select = False
    app.in_battle = False
    app.in_win = False
    app.in_lose = False
    app.canvas.fill((0, 0, 0))
    app.bg = app.guide_bg

def pressedGotIt(mousePos):
    mouseX, mouseY = mousePos
    if (370 <= mouseX <= 530 and 507 <= mouseY <= 572):
        return True
    return False

def toSelect():
    app.in_start = False
    app.in_guide = False
    app.in_select = True
    app.in_battle = False
    app.in_win = False
    app.in_lose = False
    app.canvas.fill((0, 0, 0))
    app.bg = app.select_bg

def pressedBack(mousePos):
    mouseX, mouseY = mousePos
    if (19 <= mouseX <= 100 and 540 <= mouseY <= 580):
        return True
    return False

def toBattle():
    app.in_start = False
    app.in_guide = False
    app.in_select = False
    app.in_battle = True
    app.in_win = False
    app.in_lose = False
    app.canvas.fill((0, 0, 0))
    app.bg = app.battle_bg

def toWin():
    app.in_start = False
    app.in_guide = False
    app.in_select = False
    app.in_battle = False
    app.in_win = True
    app.in_lose = False
    app.canvas.fill((0, 0, 0))
    app.bg = app.win_bg

def toLose():
    app.in_start = False
    app.in_guide = False
    app.in_select = False
    app.in_battle = False
    app.in_win = False
    app.in_lose = True
    app.canvas.fill((0, 0, 0))
    app.bg = app.lose_bg

def pressedPlayAgain(mousePos):
    mouseX, mouseY = mousePos
    if (321 <= mouseX <=578 and 456 <= mouseY <= 540):
        return True
    return False

def hoverBulb(mousePos):
    mouseX, mouseY = mousePos
    if (75 <= mouseX <= 300 and 225 <= mouseY <= 455):
        return True
    return False

def hoverChar(mousePos):
    mouseX, mouseY = mousePos
    if (375 <= mouseX <= 575 and 225 <= mouseY <= 455):
        return True
    return False

def hoverSquirt(mousePos):
    mouseX, mouseY = mousePos
    if (620 <= mouseX <= 820 and 225 <= mouseY <= 455):
        return True
    return False