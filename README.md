Clasificación de vocales con redes neuronales

Descripción del problema
Este trabajo tiene como objetivo el reconocimiento de caracteres escritos a mano, en particular de las vocales en minúscula, utilizando algoritmos simples de clasificación no lineal con redes neuronales. 
Metodología
Base de datos
La base de datos utilizada para el proyecto fue la  EMNIST: an extension of MNIST to handwritten letters” (https://arxiv.org/abs/1702.05373v1) .


La famosa base de datos MNIST se derivó de un conjunto de datos más grande conocido como NIST Special Database 19 que contiene imágenes con dimensiones de 128 x 128 pixeles de dígitos, letras escritas a mano en mayúsculas y minúsculas. Una variante del conjunto de datos NIST completo, denominado MNIST extendido (EMNIST),  sigue el mismo paradigma de conversión utilizado para crear el conjunto de datos MNIST. El resultado es un conjunto de conjuntos de datos que constituyen tareas de clasificación más difíciles que involucran letras y dígitos.

Para fines de este trabajo, se procedió a clasificar de manera manual 500 imágenes de cada vocal en minúscula, siendo un total de 2500 imágenes. 

De manera más específica, se separaron las imágenes por carpetas para cada vocal, y adicionalmente se dividió la cantidad de imágenes para entrenamiento (60%), validación (20%) y pruebas (20%). 

Por lo consiguiente, se obtuvieron 1500 imágenes para entrenamiento (300 de cada vocal), 500 para validación (100 de cada vocal) y 500 para pruebas (100 de cada vocal).  Las imágenes para entrenamiento fueron guardadas en una sola carpeta que incluyera a todas las vocales, sin embargo, se utilizó una nomenclatura de tal manera que las 300 primeras imágenes fueran correspondientes a la letra a, las siguientes 300 imágenes fueran correspondientes a la letra e, y así sucesivamente hasta llegar a la vocal u, esto con el objetivo de poder realizar posteriormente la etiquetación manual para realizar el aprendizaje supervisado. Un procedimiento similar fue realizado para las imágenes de validación y pruebas, con la diferencia de que la cantidad de imágenes por cada vocal fue de 100. 

Método
De primera instancia, para el procesamiento de las imágenes, éstas fueron reducidas a un 25% de su tamaño, es decir a 32 x 32 pixeles y convertidas a una escala de grises para así obtener una matriz de sus valores del 0 al 255, siendo 0 el color negro y el 255 el color blanco. Posteriormente, cada valor fue dividido entre 255.

Una vez realizada el procesamiento de imágenes, se procedió a realizar el etiquetado con una función en Python. Las etiquetas por cada letra corresponderían a un rango de 0 a 4, donde el número 0 representaría a la vocal a, el número 1 correspondería a la vocal e y la misma analogía para las vocales consecuentes. 

 Cada matriz de imagen fue guardada como un elemento de una lista, de acuerdo a como habían sido guardadas en su directorio correspondiente. De esta manera, los primeros 300 elementos correspondían a matrices de imágenes de la vocal a, por lo que se creó una lista alterna de etiquetas en la que los primeros 300 elementos fueran el número ‘0’. Siguiendo la analogía las siguientes etiquetas serían 300 elementos ‘1’ para la letra e, 300 elementos ‘2’ para la letra i, 300 elementos ‘3 para la letra o y finalmente 300 elementos ‘4’ para la letra u. 

De manera similar se creó una lista cuyo contenido sería la clase de cada imagen, definiendo las clases como ‘a’, ‘e’, ‘i’, ‘o’, y ‘u’.


Finalmente, una vez obtenidas las tres listas con las imágenes, etiquetas y clases, se procedió a unirlas en una sola lista, para de esta manera poder ordenar aleatoriamente los elementos y realizar la fase de entrenamiento. 


Cabe mencionar que el mismo procedimiento fue realizado para el conjunto de imágenes para validación y para las pruebas, cada una de ellas con su correspondiente etiqueta y clase. Adicionalmente, los conjuntos ordenado aleatoriamente fueron guardados en archivos para posteriormente cargarlos y procesarlos nuevamente. 

Arquitectura de la red
Para la construcción del modelo, se utilizó la biblioteca de Tensorflow Keras, una biblioteca de código abierto especializada para desarrollar modelos de aprendizaje automático. 

Esta biblioteca contiene un módulo para la implementación de modelos de redes neuronales. En este trabajo se utilizó el modelo secuencial proporcionado por la biblioteca, que es una pila lineal  de capas. 

Para la resolución de este problema, se utilizaron tres capas, una con 1024 neuronas de entrada correspondientes al número de pixeles por imagen,  otra capa con 128 neuronas con funciones de activación sigmoidales y por último, una capa de salida con 5 neuronas correspondientes a las 5 clases de vocales y con funciones de activación softmax. 

Optimización 
Para la funciones de optimización, se eligió la función “Adam”, debido a que fue la que mostró mejores resultados en la minimización del error respecto a otras funciones y la entropía cruzada categórica,  la cual mide la media de bits necesarios para identificar un evento de un conjunto de posibilidades.

Pruebas
Se realizaron un total de 500 validaciones (500 imágenes) en las cuales se presentaron resultados de precisión arriba del 90%, en específico con el 93.8% de probabilidad de acierto. Se realizaron en total 14 entrenamientos variando el número de épocas, el tamaño del lote de imágenes así como los tipos de funciones de activación tales como ‘Rectified Linear Unit (ReLU)  diferentes tipos de funciones de  optimización, tales como Stochastic Gradient Descent (SGD) y el RMSProp. En todos los casos el grado de precisión fue menor, por ejemplo al utilizar RMSProp se obtuvo un 93.2% de acierto, con ReLU un 90% y  SGD con un 88.2 %. De igual manera,  al variar el número de lotes para iterar, se notó que al disminuir el número se disminuía la precisión


