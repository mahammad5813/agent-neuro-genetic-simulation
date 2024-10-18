import pygame as pg
import pyautogui as pgui
import random
import math
import numpy as np
from nn import NN
from copy import copy, deepcopy
from nn_funcs import select_parent, crossover, select_parents, normalize, denormalize
import time
import pickle

WIDTH, HEIGHT = 800, 800
AGENT_SIZE = 20
GRAVITY = 0
tick_rate = 60
tick_limit = 200

target_radius = 10
vel_limit = 10

nn_shape = (6,50,10,2)
n_gens = 100
n_agents = 1000
n_new = 20

mut_rate = 0.1


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
        self.used_energy = 1
        # self.total_energy = 200
    
    def check_collision(self):

        if not 0<self.pos_x<WIDTH or not 0<self.pos_y<HEIGHT:
            self.dead = True
            self.score = 0



    def draw(self, win):

        pg.draw.rect(win, self.color, pg.Rect(self.pos_x, self.pos_y, AGENT_SIZE, AGENT_SIZE))

        pg.draw.rect(win, (0,0,0), pg.Rect(self.pos_x, self.pos_y, AGENT_SIZE, AGENT_SIZE), 1) #outline


    def move(self):

        if self.dead:
            return
        
        # inputs = [self.pos_x/WIDTH, self.pos_y/HEIGHT, self.vel_x/vel_limit, self.vel_y/vel_limit, self.target[0]/WIDTH, self.target[1]/HEIGHT]
        inputs = [self.pos_x/WIDTH, self.pos_y/HEIGHT, normalize(self.vel_x, -vel_limit, vel_limit), normalize(self.vel_y, -vel_limit, vel_limit), self.target[0]/WIDTH, self.target[1]/HEIGHT]



        vec = self.nn.evaluate(np.array(inputs))
        

        # self.vel_x += vec[0]*vel_limit
        # self.vel_y += vec[1]*vel_limit

        self.vel_x += denormalize(vec[0], -vel_limit, vel_limit)
        
        self.vel_y += denormalize(vec[1], -vel_limit, vel_limit)


        self.used_energy = 1+math.sqrt(vec[0]**2+vec[1]**2)
        # self.total_energy -= self.used_energy
        # print(self.used_energy)


    def check_score(self):

        dist = math.sqrt((self.pos_x+(AGENT_SIZE/2)-self.target[0])**2+(self.pos_y+(AGENT_SIZE/2)-self.target[1])**2)
        
        self.score += 1/((1+dist)**2+self.used_energy**2)
        # self.score += 1/((1+dist)**2)



    def update(self):

        if self.dead:
            return 'dead'
        
        # if self.total_energy <= 0:
        #     self.dead = True
        #     return 'dead'
        
        # print(self.total_energy)
        self.pos_x += self.vel_x
        self.pos_y += self.vel_y

        if self.vel_x:
            self.vel_x -= abs(self.vel_x)/self.vel_x*0.02
        if self.vel_y:
            self.vel_y -= abs(self.vel_y)/self.vel_y*0.02

        vel = math.sqrt(self.vel_x**2+self.vel_y**2)

        if vel > vel_limit:
            scale = vel_limit/vel
            self.vel_x*=scale
            self.vel_y*=scale

        self.check_collision()
        self.vel_y += GRAVITY
        self.check_score()


def main():
    population = [Agent(pos=[WIDTH/2, HEIGHT/2]) for _ in range(n_agents)]

    n_var = 4
    target_colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(n_var)]

    if display:
        pg.init()

        clock = pg.time.Clock()

        font = pg.font.Font(pg.font.get_default_font(), 20)
        WIN = pg.display.set_mode((WIDTH, HEIGHT))

    run_0 = True

    for gen_i in range(n_gens):
        print(gen_i)
        c_tick = 0
        
        targets = []
        n_dead = 0
        

        for i in range(n_var):
            rand_pos = [random.uniform(100, WIDTH-100), random.uniform(100, HEIGHT-100)]
            

            for agent in population[i*int(n_agents/n_var):(i+1)*int(n_agents/n_var)]:
                agent.target = rand_pos
                agent.color = target_colors[i]


            targets.append(rand_pos)


        time_start = time.time()
        run = True
        while run:
            c_tick+=1

            if c_tick>tick_limit:
                run=False

            clock.tick(tick_rate)

            if n_dead >= n_agents:
                run=False
            
            n_dead = 0
        

            for agent in population:
                if agent.dead:
                    n_dead += 1

                agent.move()
                if agent.update() == 'dead' and len(population)>=10:
                    population.remove(agent)
                # agent.update()
                    

            if display:
                keys=pg.key.get_pressed()
                if keys[pg.K_q]:
                    run = False
                    run_0 = False
                    
                # clock.tick(tick_rate)
                WIN.fill((0,0,0))

                events = pg.event.get()

                for event in events:
                    if event.type == pg.QUIT:
                        run = False


                for agent_s in population:
                    agent_s.draw(WIN)

                for i,target in enumerate(targets):
                    pg.draw.circle(WIN, target_colors[i], target, target_radius)
                    pg.draw.circle(WIN, (0,0,0), target, target_radius, 1)

                text = font.render(f"Gen: {gen_i}  Tick: {c_tick}", 1, (255,255,255))

                WIN.blit(text, (0,0))
                

                pg.display.update()

        
        if not run_0:
            break


        min_score = min([agent.score for agent in population])
        max_score = max([agent.score for agent in population])

        population.sort(key=lambda x: x.score, reverse=True)
        population = population[:int(n_agents/2)]

        for i, agent in enumerate(population):
            # population[i].score = (agent.score-min_score)/(max_score-min_score)
            population[i].score = normalize(agent.score, min_score, max_score)


        best_agents = population[:10]
        print([x.score for x in best_agents])

        print(f"Best score: {round(max_score,4)}")
        temp_population = copy(population)

        population = []
        # scores = [agent.score for agent in temp_population]
        for _ in range(int((n_agents-(n_new+10))/2)):
            parents = select_parents(temp_population)
            genes = crossover(parents[0], parents[1])
            # genes = crossover(temp_population[select_parent_njit(scores)].nn, temp_population[select_parent_njit(scores)].nn)

            # parents = [select_parent(temp_population).nn, select_parent(temp_population).nn]
            # genes = crossover_njit(parents[0].shape, parents[0].weights, parents[0].biases, parents[1].weights, parents[1].biases)
            population.append(Agent(pos=[WIDTH/2, HEIGHT/2], nn=NN(nn_shape, genes[0][0], genes[0][1])))
            population.append(Agent(pos=[WIDTH/2, HEIGHT/2], nn=NN(nn_shape, genes[1][0], genes[1][1])))

        for agent in best_agents:
            population.append(Agent(pos=[WIDTH/2, HEIGHT/2], nn=agent.nn))


        for _ in range(n_new):
            population.append(Agent(pos=[WIDTH/2, HEIGHT/2]))

        for i, agent in enumerate(population):
            rand = random.uniform(0,1)
            if rand < mut_rate:
                population[i].nn.mutate()

            population[i].score = 0
            population[i].used_energy = 1




        random.shuffle(population)


    if display:
        pg.quit()

    pop_nn = [agent.nn for agent in temp_population]
    best_agents_nn = [agent.nn for agent in best_agents]

    out_dict = {
        "population": pop_nn,
        "bests": best_agents_nn,
                }
    
    with open(f"result.pickle", "wb") as f:
        pickle.dump(out_dict, f)

if __name__ == "__main__":
    main()
