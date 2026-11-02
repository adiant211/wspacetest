YOUR TASK: write, from scratch, the weights of a policy network that plays this game as well as possible. There is no training: no data, no loss, no gradients, no fine-tuning; you choose every number yourself. You have no tools: no code execution, no calculator, nothing to run. This is a single response and you will get no feedback. First describe your design briefly (the strategy, and what the hidden units compute), then write the edits block.

THE GAME: Asterix from MinAtar (Young & Tian 2019), on a 10 x 10 grid. The player moves freely in the 4 directions. Enemies and treasure enter from the sides and move horizontally. Picking up a treasure gives +1. The episode ends when the player touches an enemy. The difficulty rises over time (enemies and treasure move faster and appear more often).
Actions: left, up, right, down move the player; fire does nothing (it acts as no-op).

The game's exact implementation (the MinAtar package, version 1.0.15, file minatar/environments/asterix.py) follows. The environment wrapper calls act(a) with a = 0..5 meaning 'n' (no-op), 'l', 'u', 'r', 'd', 'f' (this file's action_map), and with probability 0.1 it repeats the previous action instead of the new one (sticky actions). The observation the network gets is built from this file's state() as described under THE INPUT.

```python
################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################
import numpy as np


#####################################################################################################################
# Constants
#
#####################################################################################################################
ramp_interval = 100
init_spawn_speed = 10
init_move_interval = 5
shot_cool_down = 5


#####################################################################################################################
# Env
#
# The player can move freely along the 4 cardinal directions. Enemies and treasure spawn from the sides. A reward of
# +1 is given for picking up treasure. Termination occurs if the player makes contact with an enemy. Enemy and
# treasure direction are indicated by a trail channel. Difficulty is periodically increased by increasing the speed
# and spawn rate of enemies and treasure.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping=True):
        self.channels ={
            'player':0,
            'enemy':1,
            'trail':2,
            'gold':3
        }
        self.action_map = ['n','l','u','r','d','f']
        self.ramping = ramping
        self.random = np.random.RandomState()
        self.reset()

    # Update environment according to agent action
    def act(self, a):
        r = 0
        if(self.terminal):
            return r, self.terminal

        a = self.action_map[a]

        # Spawn enemy if timer is up
        if(self.spawn_timer==0):
            self._spawn_entity()
            self.spawn_timer = self.spawn_speed

        # Resolve player action
        if(a=='l'):
            self.player_x = max(0, self.player_x-1)
        elif(a=='r'):
            self.player_x = min(9, self.player_x+1)
        elif(a=='u'):
            self.player_y = max(1, self.player_y-1)
        elif(a=='d'):
            self.player_y = min(8, self.player_y+1)

        # Update entities
        for i in range(len(self.entities)):
            x = self.entities[i]
            if(x is not None):
                if(x[0:2]==[self.player_x,self.player_y]):
                    if(self.entities[i][3]):
                        self.entities[i] = None
                        r+=1
                    else:
                        self.terminal = True
        if(self.move_timer==0):
            self.move_timer = self.move_speed
            for i in range(len(self.entities)):
                x = self.entities[i]
                if(x is not None):
                    x[0]+=1 if x[2] else -1
                    if(x[0]<0 or x[0]>9):
                        self.entities[i] = None
                    if(x[0:2]==[self.player_x,self.player_y]):
                        if(self.entities[i][3]):
                            self.entities[i] = None
                            r+=1
                        else:
                            self.terminal = True

        # Update various timers
        self.spawn_timer -= 1
        self.move_timer -= 1


        #Ramp difficulty if interval has elapsed
        if self.ramping and (self.spawn_speed>1 or self.move_speed>1):
            if(self.ramp_timer>=0):
                self.ramp_timer-=1
            else:
                if(self.move_speed>1 and self.ramp_index%2):
                    self.move_speed-=1
                if(self.spawn_speed>1):
                    self.spawn_speed-=1
                self.ramp_index+=1
                self.ramp_timer=ramp_interval
        return r, self.terminal

    # Spawn a new enemy or treasure at a random location with random direction (if all rows are filled do nothing)
    def _spawn_entity(self):
        lr = self.random.rand() < 1/2
        is_gold = self.random.rand() < 1/3
        x = 0 if lr else 9
        slot_options = [i for i in range(len(self.entities)) if self.entities[i]==None]
        if(not slot_options):
            return
        slot = slot_options[self.random.randint(len(slot_options))]
        self.entities[slot] = [x,slot+1,lr,is_gold]

    # Query the current level of the difficulty ramp, could be used as additional input to agent for example
    def difficulty_ramp(self):
        return self.ramp_index

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        state = np.zeros((10,10,len(self.channels)),dtype=bool)
        state[self.player_y,self.player_x,self.channels['player']] = 1
        for x in self.entities:
            if(x is not None):
                c = self.channels['gold'] if x[3] else self.channels['enemy']
                state[x[1], x[0],c] = 1
                back_x = x[0]-1 if x[2] else x[0]+1
                if(back_x>=0 and back_x<=9):
                    state[x[1], back_x, self.channels['trail']] = 1
        return state

    # Reset to start state for new episode
    def reset(self):
        self.player_x = 5
        self.player_y = 5
        self.entities = [None]*8
        self.shot_timer = 0
        self.spawn_speed = init_spawn_speed
        self.spawn_timer = self.spawn_speed
        self.move_speed = init_move_interval
        self.move_timer = self.move_speed
        self.ramp_timer = ramp_interval
        self.ramp_index = 0
        self.terminal = False

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10,10,len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n','l','u','r','d']
        return [self.action_map.index(x) for x in minimal_actions]

```

THE INPUT (600 numbers, each 0 or 1): the 10 x 10 grid (rows 0 top to 9 bottom, columns 0 left to 9 right) with 6 channels. Input number i = (row * 10 + column) * 6 + channel; for example input 0 is (row 0, column 0, channel 0), input 6 is (row 0, column 1, channel 0) and input 599 is (row 9, column 9, channel 5). In this game:
  channel 0 (player): the player
  channel 1 (danger): enemies
  channel 2 (reward): treasure
  channel 3 (projectile): always empty in this game
  channel 4 (trail): the cell just behind each enemy and treasure (it shows which way it moves)
  channel 5 (gauge): always empty in this game
THE OUTPUT (6 numbers, logits) for the actions 0 no-op, 1 left, 2 up, 3 right, 4 down, 5 fire; the action is sampled from the softmax of the logits. With probability 0.1 the game repeats the previous action instead (sticky actions).
How the 6 channels are built from the game's own channels (state() in the source): channel 0 (player) <- player; channel 1 (danger) <- enemy; channel 2 (reward) <- gold; channel 3 (projectile) <- nothing; channel 4 (trail) <- trail; channel 5 (gauge) <- nothing. A cell is 1 in a channel if any of its source channels is 1.

THE NETWORK (the same architecture as the reinforcement-learning networks it will be compared with): a multilayer perceptron with 600 inputs, two hidden layers of 32 units with tanh activations, and 6 outputs; 20,486 parameters:
  h1 = tanh(x @ W1 + b1)          W1: 600 x 32, b1: 32
  h2 = tanh(h1 @ W2 + b2)         W2: 32 x 32,  b2: 32
  logits = h2 @ W3 + b3           W3: 32 x 6,  b3: 6
Row i of W1 holds the weights from input i to the 32 first-layer units; column j of a weight matrix belongs to unit j of the next layer. EVERY WEIGHT STARTS AT 0; you set the weights you need.

HOW YOUR NETWORK IS SCORED: its mean score per episode over 1,000 episodes, each action sampled from the softmax of the logits (with the sticky actions above); an episode also stops after 10,000 steps. It is compared with networks of the same architecture trained by reinforcement learning on this game.

HOW TO WRITE THE WEIGHTS (numbers only; nothing is run). Write a code block whose opening line is three backticks followed by the word edits, one statement per line ('#' starts a comment; a long list may continue over the next lines). Tensors: W1, b1, W2, b2, W3, b3.

```edits
W1[12, 0] = 2.5                     # one entry: row 12, column 0
W1[0:6, 3] = [1, 0, -1, 0, 2, 0]    # explicit values along a range (as many values as entries)
b1[0:4] = -1                        # fill a range (a:b is half-open; ':' alone = the whole axis)
W1[3:600:60, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]   # column 0 of channel 3, rows 0..9 of the grid -> unit 0
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]   # a 2-D list for a 2-D block (here 2 x 3)
W3[:, 2] *= 2                       # scale (also += and -=)
```
Ranges: a:b is half-open, a:b:s takes every s-th index. Input index i = (row * 10 + column) * 6 + channel. Useful strided ranges: all 10 cells of column c in channel ch are rows c*6+ch : 600 : 60 of W1; all 10 cells of row r in channel ch are rows r*60+ch : r*60+60 : 6.
Statements apply in order; a later statement overwrites an earlier one. Only the LAST edits block of your reply is read; a statement that cannot be read is skipped (the others still apply). Values: plain decimal numbers (|v| <= 1e6).