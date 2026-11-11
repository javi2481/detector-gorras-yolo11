# Trabajo Práctico - Visión por Computadora

## Entrená tu propio detector de objetos con YOLOv11! 1. Objetivos

**Objetivo general. Diseñar, implementar y evaluar un sistema de detección de objetos basado en deep learning,**

realizando fine-tuning de una red YOLOv11 pre-entrenada sobre un dataset de imágenes propio para reconocer una o más clases no presentes en el conjunto de datos COCO, analizando el impacto de las principales decisiones de diseño en la calidad final del detector.

**Objetivos específicos.**

Definir un problema de detección con una clase de interés propia del grupo, que no esté presente en el dataset COCO. Recolectar, anotar (bounding boxes) y versionar un dataset de imágenes usando Roboflow. Aplicar técnicas de data augmentation y armar la partición train / validation / test. Verificar el comportamiento out-of-the-box de un modelo pre-entrenado y constatar que la clase elegida no se detecta correctamente. Realizar fine-tuning de una YOLOv11 sobre el dataset propio y analizar las curvas de entrenamiento. Evaluar la calidad del detector de forma objetiva con métricas estándar (precision, recall, mAP@50,

**mAP@50-95).**

Comparar al menos dos configuraciones de entrenamiento distintas y discutir el efecto sobre las métricas. Realizar inferencia sobre imágenes de test no vistas y sobre imágenes "in the wild", y analizar cualitativamente aciertos y errores.

## 2. Formato del TP

**Grupos: 2 a 5 integrantes. Duración: 4 semanas. Entrega: un único notebook de Google Colab ( .ipynb ) con todo el código ejecutado y un reporte final en**

celdas de texto.

## 3. Elección del problema y del dataset

El grupo deberá elegir una (o más) clases de objetos que le resulten interesantes y que no estén bien cubiertas

**por el dataset COCO (las 80 clases de COCO son: personas, autos, perros, sillas, botellas, etc.). La gracia del TP es**

enseñarle a la red algo que todavía no sabe ver. Restricciones del dataset:

Mínimo 20 imágenes anotadas en total.

|  | Variedad real: distintos fondos, ángulos, iluminaciones, escalas y niveles de oclusión. Evitar fotos casi |
|---|---|
|  | idénticas. |
|  | Cada imagen debe tener sus bounding boxes anotadas manualmente en Roboflow. |
|  |  |
|  | Partición sugerida: 70 % train / 20 % validation / 10 % test. |
| Ideas | de clases (ver Anexo): |
|  | Objetos cotidianos regionales no presentes en COCO: un mate, un sifón, etc. |
|  |  |
|  | Logos o marcas de productos. |
|  |  |
|  | Herramientas de un taller específico. |
|  |  |
|  | Especies de plantas, flores o frutas de una región. |
|  |  |
|  | Elementos de un deporte de nicho (paletas de pádel, tabas, etc.). |
|  |  |
|  | Componentes electrónicos (resistencias, capacitores, una Raspberry Pi). |
|  |  |
|  | Señalética particular, carteles o símbolos. |
|  |  |
|  | Piezas de un juego de mesa, cartas, fichas. |
| Sean | creativos! |
|  | Importante sobre las imágenes. Usen imágenes propias o con licencia que permita su uso. No suban contenido de terceros con derechos reservados ni imágenes de personas sin su consentimiento. |
|  |  |
| 4. | Fases del trabajo |
| Fase | 1 - Dataset y anotación |
|  | Definir las clases y el criterio de anotación (¿qué cuenta como el objeto?, ¿se anotan objetos parcialmente |
|  | ocluidos o muy lejanos?). Documentar este criterio, porque condiciona toda la evaluación. |
|  | Recolectar las imágenes y subirlas a un proyecto de Roboflow. |
|  |  |
|  | Anotar las bounding boxes de cada imagen. |
|  |  |
|  | Definir la partición train / val / test. |
|  |  |
|  | Configurar al menos dos transformaciones de data augmentation (flip, rotación, brillo, blur, etc.) y justificar |
|  | por qué tienen sentido para el problema elegido. |
|  | Exportar el dataset en formato YOLOv11 y dejar el link/credenciales para descargarlo desde el notebook. |
|  |  |
|  | Reportar estadísticas: cantidad de imágenes por partición, cantidad de instancias por clase, distribución |
| Discusión casos Fase | aproximada de tamaños de los objetos. obligatoria: mostrar 3-4 imágenes anotadas de ejemplo y comentar las decisiones de anotación. ¿Qué fueron ambiguos o difíciles de anotar? 2 - Línea de base out-of-the-box |
|  | Cargar el modelo pre-entrenado yolo11n.pt (entrenado en COCO). |
|  |  |
|  | Hacer inferencia sobre 3 imágenes que contengan la clase elegida, sin ningún re-entrenamiento. |
|  |  |
|  | Mostrar los resultados y comentar: ¿la red detecta el objeto?, ¿lo confunde con alguna clase de COCO?, ¿con |

qué confianza?

**Objetivo de esta fase: documentar el punto de partida y dejar en evidencia por qué hace falta el fine-tuning.**

### Fase 3 - Fine-tuning de YOLOv11

Realizar el fine-tuning de yolo11n.pt sobre el dataset propio ( mode=train ), con plots=True . Entrenar al menos dos configuraciones distintas variando hiperparámetros, por ejemplo:

**A - Configuración base: epochs=100 , imgsz=640 . B - Variante: cambiar uno de estos factores y mantener el resto fijo, para que la comparación sea**

limpia: cantidad de épocas, tamaño de imagen ( imgsz ), tamaño del modelo ( yolo11n vs yolo11s ), o el set de data augmentation. Examinar las gráficas de entrenamiento ( results.png ): curvas de loss (box, cls, dfl), precision, recall y mAP.

**Discusión obligatoria: ¿el modelo converge?, ¿hay señales de overfitting (gap entre train y val)?, ¿qué**

configuración entrenó mejor y a qué costo?

### Fase 4 - Evaluación e inferencia

Reportar, para cada configuración, las métricas sobre el conjunto de validación: precision, recall, mAP@50 y

**mAP@50-95. Armar una tabla comparativa de las dos configuraciones.**

Ejecutar la inferencia con el mejor modelo ( best.pt ) sobre las imágenes de test (no vistas en el entrenamiento) y mostrar al menos 3 resultados. Conseguir 3 imágenes nuevas "in the wild" (de internet o propias, claramente distintas a las del dataset) y correr la inferencia sobre ellas.

**Análisis de errores obligatorio: identificar y mostrar al menos un caso de cada tipo que aparezca:**

**Verdadero positivo con buena caja. Falso positivo (detecta algo que no es). Falso negativo (no detecta un objeto presente). Caja mal ajustada (detecta pero la bounding box es imprecisa).**

## 5. Reporte final

Al final del notebook, incluir un reporte con:

1. Problema y dataset: descripción de las clases elegidas y estadísticas del dataset (un párrafo + tabla).
2. Línea de base: qué pasó al usar el modelo out-of-the-box sobre la clase elegida.
3. Resultados de entrenamiento: tabla con precision , recall , mAP@50 y mAP@50-95 para las 2 configuraciones. Análisis: ¿cuál ganó y por qué? Comentar las curvas de results.png .
4. Resultados de inferencia: mostrar resultados sobre test e imágenes "in the wild", con el análisis de errores (TP / FP / FN / cajas imprecisas).
5. Análisis de errores: ¿cuándo falla el detector? ¿hay patrones (objetos chicos, ocluidos, fondos atípicos, mala iluminación)? ¿cómo influyeron la cantidad y variedad de datos?
6. Conclusiones y trabajo futuro: qué funcionó, qué harían distinto (más datos, otra augmentation, modelo más grande), qué explorarían con más tiempo.

## 6. Entrega

**Formato: link a Colab compartido con permiso de lectura y resultados guardados para evitar correr todo el**

| notebook nuevamente. |
|---|
| Al inicio del notebook: integrantes del grupo y fecha. |
|  |
| all y verificar. Cuidar que las rutas de las carpetas de resultados ( runs/detect/predictN , |
| runs/detect/trainN ) apunten a la corrida correcta. |
| Fecha de entrega: viernes 03/07/2026 a las 23:59 h. |
|  |

## 7. Rúbrica de evaluación

| Criterio | Peso |
|---|---|
| Fase 1 - Dataset propio anotado en Roboflow, con partición, augmentation y estadísticas | 20 % |
| Fase 2 - Línea de base out-of-the-box documentada | 10 % |
| Fase 3 - Fine-tuning con dos configuraciones comparables y análisis de curvas | 20 % |
| Fase 4 - Evaluación con métricas (mAP, precision, recall) + inferencia y análisis de errores | 25 % |
| Reporte final: calidad del análisis y conclusiones fundamentadas | 15 % |
| Organización del notebook, código limpio, reproducibilidad | 10 % |

#### Causales de desaprobación:

El notebook no corre de punta a punta. Menos de tres fases implementadas, o una sola configuración de entrenamiento. Dataset que no cumple los requisitos mínimos (cantidad de imágenes, variedad, anotaciones). Clase elegida que ya está bien resuelta por COCO sin justificación. Copia de código o de datasets de otros grupos sin atribución.

## 8. Recursos recomendados

**Ultralytics YOLO11 ( ultralytics ) para entrenamiento e inferencia: https://docs.ultralytics.com Roboflow para anotar, versionar y exportar el dataset: https://roboflow.com supervision para postprocesar y visualizar detecciones. Google Colab con GPU T4 (gratuita, suficiente para todas las fases). Revisar la GPU con !nvidia-smi .**

Documentación de métricas de detección (IoU, mAP, precision-recall) para interpretar correctamente los resultados. Para exportar el modelo a uso local: yolo export model=best.pt format=onnx (u torchscript ).