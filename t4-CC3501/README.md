# tarea_04_grafica
 Tarea para el curso CC3501
Por Valeria V. Franciscangeli
vvfranciscangeli@gmail.com

- Para ver más sobre el proceso de este proyecto ingresa a:
https://github.com/valfranciscangeli/tarea_04_grafica.git

- La escena se compone de un suelo de arena, pista, dos autos, casas y barreras de contención.

- El auto que es controlado por el usuario será llamado auto 1, el auto con movimiento automático será llamado auto 2. 

- Para mover el auto 1, se utilizan las teclas ASDW o flechas:
  - A o izquierda: rotación antihoraria.
  - D o derecha: rotación horaria.
  - W o arriba: avanza.
  - S o abajo: retrocede.

- Para poder jugar, le preguntarán la velocidad con que quiere manejar el auto. Debe ser un numero positivo, o le preguntarán una y otra vez por una velocidad adecuada. Se recomienda utilizar velocidad 2.0. 
	- Para cambiar la velocidad sin reiniciar el programa, debe oprimir la tecla V y se le preguntará inmediatamente por un nuevo valor de velocidad.

- Al mover el auto 1 cambiamos la posición de la cámara, que siempre verá la escena desde la parte posterior superior del auto.

- El auto 2 se mueve automáticamente según una curva paramétrica de Bézier, formada por 4 sub-curvas. Esta curva es creada en el archivo generate_bezier.py. Este archivo se basa en la clase Auxiliar 7 y el código mostrado por Pablo Pizarro. 
	- Es posible visualizar una gráfica, los puntos y un dibujo de la curva completa que se crea. Para utilizar esto debe cambiar la variable "show" a True en la línea 6 del programa. 
	- Es importante que si cambia la variable dicha permita que se complete el dibujo o el programa no continuará. 


	