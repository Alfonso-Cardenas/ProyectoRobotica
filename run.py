from vispy import app, scene, visuals
from vispy.io import read_mesh, imread
from vispy.scene.visuals import Mesh
from vispy.visuals.transforms.linear import MatrixTransform
from vispy.visuals.transforms.chain import ChainTransform
from vispy.visuals.filters import TextureFilter
from vispy.util.quaternion import Quaternion
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from rrt import *

#---Construccion de los caminos formados por RRT---

reachedGoal = True
nodeCenters = []
nodeConnections = []
def visitTree(parent, parentIdx):
    nodeCenters.append(parent.state.p)
    currIdx = len(nodeConnections)
    if parentIdx != -1: nodeConnections.append([parentIdx, currIdx])
    for c in parent.children:
        visitTree(c, currIdx)

#------------------RRT--------------------------

print("Calculating RRT...")

p = np.array([0,0,0])
q = np.quaternion(1,0,0,0)
v = np.array([0,0,0])
w = np.array([0,0,0])

start_state = State(p, q, v, w)
start_node = Node(state = start_state)

goal_node = RRT(start_node, 10000)

centers = []
rotations = []

if(goal_node is None):
    reachedGoal = False
    print("\nNo se alcanzo la meta")
else:
    while goal_node is not None:
        centers.append(goal_node.state.p)
        rotations.append(Quaternion(*qt.as_float_array(goal_node.state.q)))
        goal_node = goal_node.parent

    centers = centers[::-1]
    translations = np.array([centers[i+1] - centers[i] for i in range(len(centers)-1)])
    rotations = rotations[::-1]
    rotations = rotations[1:]
    goodRotations = np.array([q.get_matrix().T for q in rotations])

visitTree(start_node, -1)
nodeCenters = np.array(nodeCenters)

print("\nDone calculating RRT...")

#-------------------Visual-------------------------

canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', autoswap=False, vsync=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.distance = 30
view.camera.elevation = -60.0
view.camera.azimuth = 0.0

#-----------------Caminos de RRT------------------------

line = scene.visuals.create_visual_node(visuals.LineVisual)
nodePath = line(parent = view.scene)
nodePath.set_gl_state(depth_test=True)
nodePath.set_data(pos = nodeCenters, color = [0,0,0], width = 2, connect = np.array(nodeConnections))

#-----------------Nave, asteroides y meta----------------

shipVertices, shipFaces, shipNormals, shipTexcoords = read_mesh("models/ship.obj")
shipTexture = np.flipud(imread("textures/ship.jpg"))

numOfAsteroids = len(asteroidCenters)
asteroidVertices, asteroidFaces, asteroidNormals, asteroidTexcoords = read_mesh("models/asteroid.obj")
asteroid = [Mesh(asteroidVertices, asteroidFaces, shading="smooth", color=(1,1,1)) for _ in range(numOfAsteroids)]
asteroidHitSphere = [scene.visuals.Sphere(radius=1, method='latitude', color = [1,.5,.5,.5]) for _ in range(numOfAsteroids)]

goalSquareVertices = np.array([[1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0]])@ np.diag((40, 40, 1))
goalSquareFaces = np.array([[0,1,2], [1,2,3]])
goalHitSquare = Mesh(goalSquareVertices, goalSquareFaces, color = [.55,.71,0,.5])

boxVertices = np.array([[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]]) @ np.diag((5, 2, 3.4))
boxFaces = np.array([[0,1,2], [2,1,3], [0,1,4], [4,1,5], [0,2,4], [4,2,6], [3,5,7], [5,3,1], [5,4,7], [7,4,6], [3,6,7],[6,3,2]])

shipMesh = Mesh(shipVertices, shipFaces, color=[.5,.8,.94])
hitBox = Mesh(boxVertices, boxFaces, color = [.5,.5,1,.5])

for i in range(numOfAsteroids):
    view.add(asteroid[i])
    asteroidHitSphere[i].parent = asteroid[i]
    asteroid[i].transform = MatrixTransform()
    asteroid[i].transform.translate(asteroidCenters[i])

goalHitSquare.transform = MatrixTransform()
goalHitSquare.transform.translate([0,0,goalZValue])

view.add(shipMesh)
view.add(goalHitSquare) 
hitBox.parent = shipMesh

shipTextureFilter = TextureFilter(shipTexture, shipTexcoords)
shipMesh.attach(shipTextureFilter)

rotation = MatrixTransform()
translation = MatrixTransform()

shipMesh.transform = ChainTransform([translation, rotation])

#-------------------Funcion de animacion del camino-------------------

t = 0
def update(event):
    global rotation, translation, t
    if not reachedGoal: return
    if t == 0: translation.matrix = np.eye(4)
    if(t >= len(translations)): 
        t = 0
        timer.stop()
        return
    translation.translate(translations[t])
    rotation.matrix = goodRotations[t]
    view.camera.center = centers[t]
    t += 1

# Funciones del teclado:
# Espacio    = pausa o reiniciar animacion.
# h          = hacer visible o invisible las formas con las 
#              que aproximamos nuestras mallas.
# m          = hacer visible o invisibles las mallas.
@canvas.events.key_press.connect
def on_key_press(event):
    global rotation
    global hitBox
    global asteroidHitSphere
    global shipMesh
    global asteroid
    global view
    if event.key == 'space':
        if timer.running:
            timer.stop()
        else:
            timer.start()
    elif event.key == 'h':
        hitBox.visible = not hitBox.visible
        for a in asteroidHitSphere:
            a.visible = not a.visible
    elif event.key == 'm':
        shipMesh.visible = not shipMesh.visible
        for a in asteroid:
            a.visible = not a.visible

canvas.show()

timer = app.Timer()
timer.connect(update)

if __name__ == "__main__":
    app.run()