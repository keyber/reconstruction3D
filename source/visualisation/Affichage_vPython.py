import numpy as np
from plyfile import PlyData, PlyElement
from vpython import *
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def read_ply_file(filename):
	data = PlyData.read(filename + ".ply")
	data_array = list()
	for e in data.elements[0]:
		point = list()
		for i in range(3):
			point.append(e[i])
		data_array.append(np.asarray(point))
	return np.asarray(data_array)

def draw_sphere(data, radius):
	for i in range(len(data)):
                sphere(radius=radius[i], pos=vector(data[i][0], data[i][1], data[i][2]), color=color.yellow, emissive=True)

        
def radius_estimation(data):
	matrix_distances = squareform(pdist(data))
				
	radius_array = np.zeros(len(data))
	for i in range(len(matrix_distances)):
		radius_array[i]	= np.min(matrix_distances[i][np.where(matrix_distances[i] != 0)])
			
	return radius_array
                
                
def main():
        file_name = "1a04e3eab45ca15dd86060f189eb133.points"
        data = read_ply_file(file_name)
        radius = radius_estimation(data[:100,:])
        scene = canvas(title="Spheres", width=1200, height=900)
        scene.forward = vector(0,0,0)
        scene.userzoom = True
        scene.userspin = True
        scene.range = 2
        l1 = local_light(pos=vector(0,0,0), color=color.white)
        draw_sphere(data[:100,:],radius)
        zoom = False
        spin = False
        while True:
                rate(50)
                #scene.center = rotate(scene.forward, angle=4*arcsin(0.5), axis=right)
                # if scene.kb.keys:
                #         k = scene.kb.getkey()
                #         if k == 'f':
                #                 scene.range = 5
                #         elif k == 'g':
                #                 scene.range = 3
                # if zoom:
                #         newy = scene.mouse.pos.y
                #         if newy != lasty:
                #                 distance = (scene.center-scene.mouse.camera).mag
                #                 scaling = 10**((lasty-newy)/distance)
                #                 newrange = scaling*scene.range.y
                #                 if rangemin < newrange < rangemax:
                #                         scene2.range = newrange
                #                         lasty = scaling*newy
                # elif spin:
                #         newray = scene.mouse.ray
                #         dray = newray-lastray
                #         right = scene.forward.cross(scene.up).norm() # unit vector to the right
                #         up = right.cross(scene.forward).norm() # unit vector upward
                #         anglex = -4*arcsin(dray.dot(right))
                #         newforward = vector(scene.forward)
                #         newforward = rotate(newforward, angle=anglex, axis=scene.up)
                #         newray = rotate(newray, angle=anglex, axis=scene.up)
                #         angley = 4*arcsin(dray.dot(up))
                #         maxangle = scene.up.diff_angle(newforward)
                #         if not (angley >= maxangle or angley <= maxangle-pi):
                #                 newforward = rotate(newforward, angle=angley, axis=right)
                #                 newray = rotate(newray, angle=angley, axis=right)
                #         scene.forward = newforward
                #         lastray = newray


	
main()
