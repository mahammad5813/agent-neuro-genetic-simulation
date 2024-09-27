import pygame as pg
import pyautogui as pgui
import random
import math
import numpy as np
from nn import NN
from copy import copy, deepcopy
from nn_funcs import select_parent, crossover
import time

WIDTH, HEIGHT = 800, 600
AGENT_SIZE = 20
GRAVITY = 0
tick_rate = 60
tick_limit = 200

nn_shape = (6,100,50,2)
n_gens = 100
n_agents = 1000

n_new = 20


display = True


class Agent:
    def __init__(self, pos: list[float] = None, vel: list[float]=None, nn: NN = None, target: list[float:2]=None, color = None):
        if not pos:
            pos = [WIDTH/2,HEIGHT/2]

        self.pos_x = pos[0]
        self.pos_y = pos[1]

        if not vel:
            vel = [0,0]
        self.vel_x = vel[0]
        self.vel_y = vel[1]

        if nn:
            self.nn = nn
        else:
            self.nn = NN(nn_shape)

        if target:
            self.target = target
        else:
            self.target = [random.uniform(0, WIDTH), random.uniform(0, HEIGHT)]

        if color:
            self.color = color
        else:
            self.color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        self.score = 0
        self.dead = False
    
    def check_collision(self):

        if not 0<self.pos_x<WIDTH:
            self.dead = True
            self.score = 0
            return
        if not 0<self.pos_y<HEIGHT:
            self.dead = True
            self.score = 0


    def draw(self, win):
        pg.draw.rect(win, self.color, pg.Rect(self.pos_x, self.pos_y, AGENT_SIZE, AGENT_SIZE))
        pg.draw.rect(win, (0,0,0), pg.Rect(self.pos_x, self.pos_y, AGENT_SIZE, AGENT_SIZE), 1) #outline


    def move(self):
        if self.dead:
            return
        
        inputs = [self.pos_x, self.pos_y, self.vel_x, self.vel_y, self.target[0], self.target[1]]


        vec = self.nn.evaluate(np.array(inputs))
        vec/=200

        self.pos_x += vec[0]
        self.pos_y += vec[1]


    def check_score(self):
        dist = math.sqrt((self.pos_x+(AGENT_SIZE/2)-self.target[0])**2+(self.pos_y+(AGENT_SIZE/2)-self.target[1])**2)
        self.score += 1/((1+dist)**2)


    def update(self):
        if self.dead:
            return
        
        self.pos_x += self.vel_x
        self.pos_y += self.vel_y
        if self.vel_x:
            self.vel_x -= abs(self.vel_x)/self.vel_x*0.02
        if self.vel_y:
            self.vel_y -= abs(self.vel_y)/self.vel_y*0.02
        self.check_collision()
        self.vel_y += GRAVITY
        self.check_score()



population = [Agent(pos=[WIDTH/2, HEIGHT/2]) for _ in range(100)]


clock = pg.time.Clock()

if display:
    pg.init()

    font = pg.font.Font(pg.font.get_default_font(), 20)
    WIN = pg.display.set_mode((WIDTH, HEIGHT))

run_0 = True

for gen_i in range(n_gens):

    c_tick = 0

    targets = []
    target_colors = []


    for i in range(4):
        rand_pos = [random.uniform(100, WIDTH-100), random.uniform(100, HEIGHT-100)]
        rand_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

        for agent in population[i*int(n_agents/4):(i+1)*int(n_agents/4)]:
            agent.target = rand_pos
            agent.color = rand_color


        targets.append(rand_pos)
        target_colors.append(rand_color)
    random.shuffle(population)


    time_start = time.time()
    run = True
    while run:
        c_tick+=1
        if c_tick>tick_limit:
            run=False

        # clock.tick(tick_rate)


        for agent_s in population:
            agent_s.update()
            agent_s.move()
                

        if display:
            keys=pg.key.get_pressed()
            if keys[pg.K_q]:
                run = False
                run_0 = False
                
            clock.tick(tick_rate)
            WIN.fill((0,0,0))

            events = pg.event.get()

            for event in events:
                if event.type == pg.QUIT:
                    run = False


            for agent_s in population:
                agent_s.draw(WIN)

            for i,target in enumerate(targets):
                pg.draw.circle(WIN, target_colors[i], target, 10)
                pg.draw.circle(WIN, (0,0,0), target, 10, 1)

            text = font.render(f"Gen: {gen_i}  Tick: {c_tick}", 1, (255,255,255))

            WIN.blit(text, (0,0))
            

            pg.display.update()
    
    if not run_0:
        break

    population.sort(key=lambda x: x.score, reverse=True)
    population = population[:int(n_agents/2)]

    min_score = min([agent.score for agent in population])
    max_score = max([agent.score for agent in population])

    for i, agent in enumerate(population):
        agent.score = (agent.score-min_score)/(max_score-min_score)

    bests = population[:10]
    print(f"Previous best: {round(max_score,4)}")
    temp_population = copy(population)

    population = []

    for _ in range(int((n_agents-(n_new + 10))/2)):
        genes = crossover(select_parent(temp_population), select_parent(temp_population))

        population.append(Agent(pos=[WIDTH/2, HEIGHT/2], nn=NN(nn_shape, genes[0][0], genes[0][1])))
        population.append(Agent(pos=[WIDTH/2, HEIGHT/2], nn=NN(nn_shape, genes[1][0], genes[1][1])))

    for best in bests:
        population.append(Agent(pos=[WIDTH/2, HEIGHT/2], nn=best.nn))


    for _ in range(n_new):
        population.append(Agent(pos=[WIDTH/2, HEIGHT/2]))

    for agent in population:
        agent.nn.mutate()
        agent.score = 0



    random.shuffle(population)


if display:
    pg.quit()
