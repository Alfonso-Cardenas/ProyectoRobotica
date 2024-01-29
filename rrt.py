import numpy as np
from numpy import linalg as la
import quaternion as qt
import random
import sys

UF = np.array([[0,0,1], [0,0,-1]])
UT = np.array([[0.02,0,0], [-0.02,0,0], [0,0.02,0], [0,-0.02,0], [0,0,0.02], [0,0,-0.02]]) #Control para nube de asteroides
#UT = np.array([[0,0,-.1], [0,0,.1]]) #Control para mapa chico
step = 0.5

HITBOX_MIN = np.array([-5, -2, -3.4])
HITBOX_MAX = np.array([5, 2, 3.4])

spaceshipLength = 6.8

#Generacion de centros de asteroides en mapa chico
# asteroidCenters = np.array([
#     [0,3.5,2*spaceshipLength], [0,-3.5,2*spaceshipLength],
#     [3.5,0,4*spaceshipLength], [-3.5,0,4*spaceshipLength],
#     [0,3.5,6*spaceshipLength], [0,-3.5,6*spaceshipLength],
#     ]) 

asteroidRadius = 1

goalZValue = 7*spaceshipLength

x_min, y_min, z_min = -20, -20, 0
x_max, y_max, z_max = 20, 20, 12*spaceshipLength

spaceship_min = [x_min, y_min, z_min]
spaceship_max = [x_max, y_max, z_max]

def box_intersects_ball(r, c, b_min, b_max):
    p = c.copy()
    p[c < b_min] = b_min[c < b_min]
    p[c > b_max] = b_max[c > b_max]
    if np.linalg.norm(p - c) <= r:
        return True
    return False

#Generacion de centros de asteroides en mapa chico
numOfAsteroids = 40
asteroidCenters = np.zeros((numOfAsteroids, 3))
i = 0
while i < numOfAsteroids:
    asteroidCenters[i] = [np.random.uniform(x_min, x_max),
                         np.random.uniform(y_min, y_max),
                         np.random.uniform(z_min, goalZValue)]
    if box_intersects_ball(asteroidRadius, asteroidCenters[i], HITBOX_MIN, HITBOX_MAX):
        continue
    i += 1

print("Done calculating map!")

def random_unit_quaternion():
    # Generate random axis
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)

    # Generate random angle
    angle = np.random.uniform(0, 2 * np.pi)

    # Create quaternion from axis and angle
    q = qt.from_rotation_vector(axis * angle)

    return q / np.abs(q) 

class Control:
    def __init__(self, F = None, T = None):
        if(F is not None):
            self.F = F
            self.T = T
        else:
            self.F = np.zeros(3)
            self.T = np.zeros(3)
           
            
    def toState(self, prevState):
        Rt = qt.as_rotation_matrix(prevState.q)
        
        
        v_point  = Rt @ (self.F / 10)
        w_point  = self.T
        v_new = prevState.v + (step * v_point)
        w_new = prevState.w + (step * w_point)
        
        
        p_point = v_new
        q_point  = 0.5 * (np.quaternion(0.0 , w_new[0], w_new[1], w_new[2]) * prevState.q)
        
        p_new = prevState.p + (step * p_point)
        q_new = prevState.q + (step * q_point)
        q_new = np.normalized(q_new)
        
        return State(p_new, q_new, v_new, w_new)
        
class State:
    def __init__(self, p = None, q = None, v = None, w = None):
        if(p is not None):
            self.p = p
            self.q = q
            self.v = v
            self.w = w
        else:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            z = random.uniform(z_min, z_max)
            self.p = np.array([x, y, z])
            self.q = random_unit_quaternion()
            self.v = np.random.randn(3)
            self.w = np.random.randn(3)
            return
        
    def distanceToState(self, state):
        p_weight, q_weight, v_weight, w_weight = 1, 1, 1, 1
        return ( p_weight * (np.dot(self.p - state.p, self.p - state.p)) + q_weight * ((1 - np.abs(self.q * state.q)) ** 2) 
                + v_weight * (np.dot(self.v - state.v, self.v - state.v)) + w_weight * (np.dot(self.w - state.w, self.w - state.w)))
    
    def nearestNode(self, root):
        if (root is None):
            return None
        nearest = root
        min_distance = self.distanceToState(root.state)
        for child in root.children:
            child_nearest = self.nearestNode(child)
            child_min_distance = self.distanceToState(child_nearest.state)
            
            if(child_min_distance < min_distance):
                nearest = child_nearest
                min_distance = child_min_distance
        
        return nearest
    
    def detectCollision(self):
        if np.any(self.p < spaceship_min): return True
        if np.any(self.p > spaceship_max): return True

        Rt = qt.as_rotation_matrix(self.q.conjugate())

        for c in asteroidCenters:
            transformed_c = Rt @ (c - self.p)
            if(box_intersects_ball(asteroidRadius, transformed_c, HITBOX_MIN, HITBOX_MAX)):
                return True
        
        return False

    def stateReachedGoal(self):
        return self.p[2] > goalZValue
    
class Node:
    def __init__(self, state = None, control = None, parent = None):
        self.state = state
        self.children = []
        if(control is not None):
            self.control = control
            self.parent = parent
        else:
            self.control = None
            self.parent = None
    
    def insertNewNode(self, randomState):
        min_distance = sys.float_info.max
        min_control = self.control
        min_state = self.state

        np.random.shuffle(UF)
        np.random.shuffle(UT)
       
        for F in UF:
            for T in UT:
                new_control = Control(F, T)
                new_state = new_control.toState(self.state)
                
                if(not new_state.detectCollision()):
                    new_distance = randomState.distanceToState(new_state)
                    if( new_distance < min_distance):
                        min_distance = new_distance
                        min_control = new_control
                        min_state = new_state
        
        if(min_distance != sys.float_info.max):
            new_node = Node(min_state, min_control, self)
            self.children.append(new_node)
            return True
        
        return False
    
    def nodeReachedGoal(self):
        return self.state.stateReachedGoal()
    
def RRT(start, max_K):
    for i in range(max_K):
        print("Current iteration = ",i, end="\r")
        rand_state = State()
        nearest_node = rand_state.nearestNode(start)
        
        if(nearest_node.insertNewNode(rand_state)):
            new_node = nearest_node.children[-1]
            if(new_node.nodeReachedGoal()):
                return new_node
            
    
    return None