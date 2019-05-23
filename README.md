#Clasificación de vocales con redes neuronales

Descripción del problema
Este trabajo tiene como objetivo el reconocimiento de caracteres escritos a mano, en particular de las vocales en minúscula, utilizando algoritmos simples de clasificación no lineal con redes neuronales. 
Metodología
Base de datos
La base de datos utilizada para el proyecto fue la  EMNIST: an extension of MNIST to handwritten letters” (https://arxiv.org/abs/1702.05373v1) .


La famosa base de datos MNIST se derivó de un conjunto de datos más grande conocido como NIST Special Database 19 que contiene imágenes con dimensiones de 128 x 128 pixeles de dígitos, letras escritas a mano en mayúsculas y minúsculas. Una variante del conjunto de datos NIST completo, denominado MNIST extendido (EMNIST),  sigue el mismo paradigma de conversión utilizado para crear el conjunto de datos MNIST. El resultado es un conjunto de conjuntos de datos que constituyen tareas de clasificación más difíciles que involucran letras y dígitos.

Para fines de este trabajo, se procedió a clasificar de manera manual 500 imágenes de cada vocal en minúscula, siendo un total de 2500 imágenes. 

