import numpy as np
import random
import pygame
from pygame import *
from OpenGL.GL import *
from OpenGL.GLU import *


def reconstrution_faces(filename, mlp, grid):
	"""
	Fonction pour reconstuire les faces de l'objet à partir du dernier nuage d'animation et de la grille
	:param filename: nom du fichier contenant le nuages de points de l'objet
	:param mlp: nombre de mlp
	:param grid: taille de la grille
	:return: les faces reconstruites et les couleurs à appliquer à l'objet
	"""
	data = np.load(filename + ".npy")
	data = data[-1]  # Récupérer le dernier nuage de l'animation
	data = data.reshape((mlp, grid, grid, 3)) 
	faces = list()
	for i in range(data.shape[0]):
		for j in range(data.shape[1]-1):
			for k in range(data.shape[2]-1):
				faces.append((data[i,j,k], data[i,j+1,k], data[i,j+1,k+1], data[i,j,k+1])) # Reconstruction d'une face

	# Génération des couleurs
	colors = [(random.randrange(0, 10) / 10, random.randrange(0, 10) / 10, random.randrange(0, 10) / 10) for i in range(4)]
	return faces, colors


def draw_faces(faces, colors):
	"""
	Fonction pour dessiner les faces de l'objet
	:param faces: les faces de l'objet en question
	:param colors: les couleurs à lui appliquer
	"""
	glBegin(GL_QUADS)
	for surface in faces:
		c = 0
		for vertex in surface:
			glColor3fv(colors[c])
			c += 1
			glVertex3fv(vertex)

	glEnd()


def drawing_shape(filename, mlp, grid):

	"""
	Fonction pour dessiner l'objet reconstruit
	:param filename: nom du fichier contenant le nuages de points de l'objet
	:param mlp: nombre de mlp
	:param grid: taille de la grille
	:return: une fênetre affichant l'objet et une image de l'objet portant le même nom du fichier
	"""
	f, c = reconstrution_faces(filename, mlp, grid)

	glEnable(GL_DEPTH_TEST)
	glDepthFunc(GL_ALWAYS)
	pygame.init()
	display = (800, 600)
	s = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

	gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
	glTranslatef(0.0, 0.0, -5.0)
	glRotatef(0, 0, 0, 0)

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
				exit()

		glRotatef(1, 3, 1, 1)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		draw_faces(f, c)
		pygame.image.save(s, filename+".jpeg") # Sauvegarder une image de l'objet rencontruit à la fin de l'animation
		pygame.display.flip()
		pygame.time.wait(10)
