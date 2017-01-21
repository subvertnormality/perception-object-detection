import pygame as pg
import time

pg.mixer.init()
pg.init()

pg.mixer.set_num_channels(50)

def one():
  one = pg.mixer.Sound("../sounds/1.wav")
  one.play(loops = 0)

def two():
  two = pg.mixer.Sound("../sounds/2.wav")
  two.play(loops = 0)

def three():
  three = pg.mixer.Sound("../sounds/3.wav")
  three.play(loops = 0)

def level_complete():
  three = pg.mixer.Sound("../sounds/level_complete.wav")
  three.play(loops = 0)