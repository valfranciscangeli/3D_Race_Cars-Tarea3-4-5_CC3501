# tarea_05_graficas
Tarea para el curso CC3501
Por Valeria V. Franciscangeli
vvfranciscangeli@gmail.com

- Para ver más sobre el proceso de este proyecto ingresa a:
https://github.com/valfranciscangeli/tarea_05_graficas.git

- La escena se compone de un suelo de arena, pista, dos autos, casas y barreras de contención.

- El auto que es controlado por el usuario será llamado auto 1 (controlado), el auto con movimiento automático será llamado auto 2 (Bezier/ automático). El auto 1 parte 1.5 unidad en z más atrás que el auto 2.

- Para mover el auto 1, se utilizan las teclas ASDW o flechas:
  - A o izquierda: rotación antihoraria.
  - D o derecha: rotación horaria.
  - W o arriba: avanza.
  - S o abajo: retrocede.

- Para poder jugar, le preguntarán la velocidad con que se mueva el auto2 y la velocidad para manejar el auto 1. Debe ser un número positivo entero entre de 1 a 3, o se activará la velocidad por defecto (nivel 1). El nivel 3 es la velocidad máxima. 
	- Para cambiar la velocidad del auto 1 sin reiniciar el programa, debe oprimir la tecla V y se le preguntará inmediatamente por un nuevo valor de velocidad.
	- No se puede cambiar la velocidad del auto 2 sin reiniciar el programa. 

- Al mover el auto 1 cambiamos la posición de la cámara, que siempre verá la escena desde la parte posterior superior del auto.
- El auto 2 se mueve automáticamente según una curva paramétrica de Bézier.

- Se implementan AABB en la escena para analizar las colisiones, utilizando la clase AABB y AABBlist creada por Sebastián Olmos para la Clase Auxiliar 11- Colisiones. 
    - Las AABB implementadas en los autos son dinámicas, es decir, se busca rodear el auto completamente independiente de su posición, llegando a sus mayores dimensiones cuando los autos se encuentran en diagonal (45°). 
    - Para dejar de chequear colisiones debe oprimir el botón ENTER (para volver a chequear: lo mismo). 
    - Para ver (o no) las AABB en escena debe oprimir LEFT CTRL. 
    - Al detectarse que el auto 1 colisiona con algún objeto del mundo (casas y muros) o auto2, no se permite que se mueva hasta que se intente otra dirección. 
    - Al detectarse colisión de auto 2 con el mundo (caso que nunca ocurrirá, ya que sigue la ruta de la curva de Bezier) o con el auto 1, queda detenido.
    



