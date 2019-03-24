import pygame
from pygame import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Defining a list of vertices
vertices = (
(1,-1,-1),
(1,1,-1),
(-1,1,-1),
(-1,-1,-1),
(1,-1,1),
(1,1,1),
(-1,-1,1),
(-1,1,1)
)

# Defining a list a surfaces
surfaces = (
(0,1,2,3),
(3,2,7,6),
(6,7,5,4),
(4,5,1,0),
(1,5,7,2),
(4,0,3,6)
)

# Defining a list of colors
colors = (
(0,0,0),
(1,0,0),
(0,1,0),
(0,0,1),
(1,0,0)
)
# Function to draw the cube
def Cube():	
	glBegin(GL_QUADS)
	for surface in surfaces:
		color = 0
		for vertex in surface:
			color += 1
			glColor3fv(colors[color])
			glVertex3fv(vertices[vertex])
	glEnd()

	
def main():

	pygame.init()
	display = (800,600)
	pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
	
	gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
	glTranslatef(0.0, 0.0, -5.0)
	glRotatef(0, 0, 0, 0)
	
	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
				exit()
				
		glRotatef(1, 1, 1, 1)		
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		Cube()
		pygame.display.flip()
		pygame.time.wait(10)

		
if __name__ == '__main__':
	main()
		
	
	
	
	
	
	
