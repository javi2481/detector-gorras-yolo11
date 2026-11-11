# Detector de gorras con YOLO11

Fine-tuning de YOLO11 para detectar **gorras con visera**, una clase que el modelo pre-entrenado en COCO no conoce.

| | |
|---|---|
| **Institución** | ISTEA |
| **Materia** | Aprendizaje Automático II |
| **Año / cuatrimestre** | 2026 — 2.º cuatrimestre |
| **Equipo** | Mauricio Chocholacek · Paola Lucia Pierolivo Fernandez · Veronica Andrea Gonzalez · Daiana Marilyn Ruiz Díaz · Andrea Romeo · Rodolfo Javier Berrone |
| **Entrega** | Trabajo Práctico 2 — Visión por Computadora · 13/11/2026 |

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/javi2481/detector-gorras-yolo11/blob/main/notebooks/TP2_Detector_Gorras.ipynb)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.210-111F68)
![YOLO11](https://img.shields.io/badge/modelo-YOLO11n%20%7C%20YOLO11s-00A67E)
[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-yellow)](LICENSE)

## Contenido

- [El proyecto en una imagen](#el-proyecto-en-una-imagen)
- [Cómo funciona](#cómo-funciona)
- [1. Los datos](#1-los-datos)
- [2. Línea de base](#2-línea-de-base-el-modelo-sin-entrenar)
- [3. Entrenamiento: A vs. B](#3-entrenamiento-a-vs-b)
- [4. Prueba final y análisis de errores](#4-prueba-final-y-análisis-de-errores)
- [Qué aprendimos](#qué-aprendimos)
- [Reproducir](#reproducir)
- [Estructura del repositorio](#estructura-del-repositorio)
- [El notebook, sección por sección](#el-notebook-sección-por-sección)
- [Equipo](#equipo)
- [Glosario rápido](#glosario-rápido)
- [Créditos](#créditos)
- [Licencia](#licencia)

## El proyecto en una imagen

![Progreso del proyecto](figuras/01_progreso.png)

Con 64 fotos propias, pasamos de un modelo que **no veía ninguna gorra** a uno que encuentra **6 de 7** en test. Pero con fotos realmente nuevas encuentra **1 de 3**. Este README explica cómo llegamos a cada número y qué aprendimos de esa diferencia.

## Cómo funciona

```mermaid
flowchart TD
    subgraph S1["1. Dataset"]
        A[("Roboflow<br/>64 fotos · 72 gorras")]
        B["Partición 45 / 13 / 6<br/>aumentación ×4 solo en train"]
    end

    subgraph S2["2. Línea de base"]
        C["yolo11n.pt sin entrenar<br/>0 de 3 gorras detectadas"]
    end

    subgraph S3["3. Fine-tuning"]
        D["A: yolo11n<br/>100 épocas · 640 px"]
        E["B: yolo11s<br/>100 épocas · 640 px"]
    end

    subgraph S4["4. Elegir ganador"]
        F{"mAP@50-95<br/>sobre valid"}
        G[/"best.pt de A"/]
    end

    subgraph S5["5. Prueba final"]
        H["Test<br/>6 de 7 gorras"]
        I["In the wild<br/>1 de 3"]
        J["Análisis de errores<br/>TP · FP · FN · caja mal ajustada"]
    end

    A --> B
    B -.->|"3 fotos aparte"| C
    B --> D
    B --> E
    D --> F
    E --> F
    F -->|"gana A: 0.698 vs 0.655"| G
    G ==> H
    G ==> I
    H --> J
    I --> J

    classDef datos fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef base fill:#f3f4f6,stroke:#6b7280,color:#1f2937
    classDef train fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef elegir fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef prueba fill:#fce7f3,stroke:#db2777,color:#831843

    class A,B datos
    class C base
    class D,E train
    class F,G elegir
    class H,I,J prueba

    style S1 fill:#eff6ff,stroke:#2563eb,color:#1e3a8a
    style S2 fill:#f9fafb,stroke:#6b7280,color:#1f2937
    style S3 fill:#f0fdf4,stroke:#16a34a,color:#14532d
    style S4 fill:#fffbeb,stroke:#d97706,color:#78350f
    style S5 fill:#fdf2f8,stroke:#db2777,color:#831843
```

| Etapa | Pregunta que responde | Resultado |
|---|---|---|
| **1. Dataset** (azul) | ¿Qué le enseñamos? | 64 fotos, 72 gorras, una sola clase |
| **2. Línea de base** (gris) | ¿Hacía falta entrenar? | Sí: sin entrenar no detecta ninguna gorra |
| **3. Fine-tuning** (verde) | ¿Qué variante aprende mejor? | Dos modelos que solo difieren en el tamaño |
| **4. Elegir ganador** (amarillo) | ¿Con cuál seguimos? | A (`yolo11n`): ajusta mejor las cajas y pesa un cuarto |
| **5. Prueba final** (rosa) | ¿Funciona con fotos que no vio? | Test 6 de 7 · in the wild 1 de 3 |

## 1. Los datos

```mermaid
flowchart TD
    A[("64 fotos propias<br/>72 gorras marcadas")]
    A --> B["Train · 45 fotos<br/>📘 estudiar"]
    A --> C["Valid · 13 fotos<br/>📝 simulacro: elegir A o B"]
    A --> D["Test · 6 fotos<br/>🎓 examen final"]
    B -->|"aumentación ×4<br/>flip H/V · rotación ±18° · ruido"| E["180 fotos de entrenamiento"]
    F["3 fotos in the wild<br/>🌍 vida real, fuera del dataset"]

    classDef origen fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef train fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef valid fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef test fill:#fce7f3,stroke:#db2777,color:#831843
    classDef wild fill:#f3e8ff,stroke:#9333ea,color:#581c87

    class A origen
    class B,E train
    class C valid
    class D test
    class F wild
```

Cada grupo de fotos cumple un papel distinto, como un estudiante preparando un examen: con **train** estudia, con **valid** hace el simulacro (lo usamos para elegir entre A y B), **test** es el examen final con ejercicios del mismo libro, e **in the wild** es un problema de la vida real.

Las gorras se marcaron a mano en Roboflow con un criterio común: solo gorras con visera, un recuadro por gorra, y se marcan las tapadas o cortadas si se ve la mayor parte.

| Partición | Fotos originales | Imágenes en el export | Para qué se usa |
|---|---:|---:|---|
| Train | 45 | 180 (×4 por aumentación) | Entrenar |
| Valid | 13 | 13 | Elegir entre A y B |
| Test | 6 | 6 | Evaluación final |
| In the wild | 3 | — (fuera del dataset) | Probar con fotos realmente nuevas |

**Preprocesamiento:** orientación automática y *resize* (estirado) a 640x640. Las aumentaciones se aplican solo a train; valid y test quedan sin aumentar para evaluar con fotos reales.

![Ejemplos anotados](figuras/02_ejemplos_anotados.png)

## 2. Línea de base: el modelo sin entrenar

`yolo11n.pt`, entrenado en COCO, ve la silla, la botella y las plantas, pero **ninguna gorra**: la ignora o la llama "person".

![Línea de base](figuras/03_linea_base.png)

## 3. Entrenamiento: A vs. B

Se compararon dos configuraciones que cambian **un solo factor**, el tamaño del modelo. Todo lo demás es igual: 100 épocas, 640 px y semilla 42.

```mermaid
flowchart LR
    D[("Mismo dataset<br/>mismos hiperparámetros<br/>100 épocas · 640 px · seed 42")]
    D --> A["A — yolo11n<br/>2.6 M parámetros"]
    D --> B["B — yolo11s<br/>9.4 M parámetros"]
    A --> MA["mAP@50-95 = 0.698"]
    B --> MB["mAP@50-95 = 0.655"]
    MA --> G{{"Gana A"}}
    MB --> G

    classDef datos fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef a fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef b fill:#f3f4f6,stroke:#6b7280,color:#1f2937
    classDef win fill:#fef3c7,stroke:#d97706,color:#78350f

    class D datos
    class A,MA a
    class B,MB b
    class G win
```

| Configuración | Precision | Recall | mAP@50 | mAP@50-95 | Tiempo (T4) | Tamaño |
|---|---:|---:|---:|---:|---:|---:|
| **A — yolo11n (ganador)** | 0.987 | 0.867 | 0.923 | **0.698** | 5.5 min | 5.5 MB |
| B — yolo11s | 0.982 | 0.867 | **0.970** | 0.655 | 6.3 min | 19.2 MB |

![Comparación de curvas A vs B](figuras/05_comparacion_A_B.png)

<details>
<summary>Ver el <code>results.png</code> completo de A (todas las pérdidas y métricas)</summary>

![results.png de A](figuras/04_results_A.png)

</details>

- **Las dos convergen y ninguna sobreajusta:** las pérdidas de validación se aplanan y no vuelven a subir.
- **B encuentra algo mejor las gorras** (mAP@50), **pero A ajusta mejor las cajas** (mAP@50-95, la métrica que elegimos como principal).
- **Con 45 fotos de entrenamiento, el modelo grande no tiene con qué aprovechar su capacidad extra.** Gana el más barato.

## 4. Prueba final y análisis de errores

Cada caja que predice el modelo se compara con el recuadro real usando **IoU**: el área en común dividida por el área total de las dos cajas.

```mermaid
flowchart TD
    P["Cada caja que predice el modelo"] --> Q{"¿Cuánto se superpone (IoU)<br/>con una gorra real todavía libre?"}
    Q -->|"IoU ≥ 0.75"| TP["✅ Acierto con buena caja"]
    Q -->|"0.3 ≤ IoU < 0.75"| MA["⚠️ Caja mal ajustada"]
    Q -->|"IoU < 0.3"| FP["❌ Falso positivo<br/>detectó algo que no es"]
    R["Cada gorra real que<br/>ninguna caja encontró"] --> FN["❌ Falso negativo<br/>se le escapó"]

    classDef ok fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef medio fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef mal fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    classDef neutro fill:#f3f4f6,stroke:#6b7280,color:#1f2937

    class TP ok
    class MA medio
    class FP,FN mal
    class P,Q,R neutro
```

| | Test (7 gorras) | Valid (15 gorras) |
|---|---:|---:|
| ✅ Acierto con buena caja | 6 | 12 |
| ⚠️ Caja mal ajustada | 0 | 2 |
| ❌ Falso negativo | 1 | 1 |
| ❌ Falso positivo | 5 | 2 |

El detalle de cada predicción está en [`resultados/casos_test_valid.csv`](resultados/casos_test_valid.csv).

Dos fotos de test, con el recuadro real a la izquierda y la predicción a la derecha. Arriba aparecen dos falsos positivos sobre el borde oscuro de la tele; abajo, el modelo pierde la gorra gris clara:

![Test: recuadro real vs predicción](figuras/06_test_real_vs_prediccion.png)

Un ejemplo de cada tipo (verde: recuadro real · rojo: predicción):

![Ejemplos de cada tipo de error](figuras/07_ejemplos_errores.png)

**In the wild:** tres fotos de exterior con sol fuerte y la gorra colgada. Acertó en 1 de 3.

![Fotos in the wild](figuras/08_in_the_wild.png)

| Foto | Original (conf 0.25) | Estirada a 640x640 |
|---|---|---|
| `gorra_2d` | Falso negativo | Sigue sin detectar la gorra, más un falso positivo (0.41) |
| `gorra_5d` | Falso negativo | Sigue sin detectar la gorra, más 2 falsos positivos |
| `gorra_6d` | **Acierto** (0.80) | Acierto (0.85), más un falso positivo |

## Qué aprendimos

El detector funciona bien con fotos parecidas a las de entrenamiento y falla fuera de ellas. Encontramos cinco patrones:

1. **Oscuro = gorra.** La mayoría de las gorras del dataset son negras. Los falsos positivos caen sobre zonas oscuras (el borde de una tele, una sombra, hojas), y la gorra que se perdió es la gris clara.
2. **Pose.** Las gorras del dataset están casi siempre apoyadas. La gorra in the wild también es negra, pero colgada: no la detecta.
3. **El test es optimista.** La partición se hizo por foto, no por gorra, así que las mismas gorras aparecen en train y en test (la gorra floreada de test está en 8 fotos de train). Además, test viene estirado a 640x640 igual que train; las fotos nuevas no.
4. **Sensibilidad al umbral.** Una gorra chica entró con 0.29 de confianza (el umbral es 0.25) y los falsos positivos van de 0.32 a 0.66: un cambio chico de umbral cambia los resultados.
5. **Gorras cortadas por el borde** dan cajas mal ajustadas.

> **La precision de 0.987 no contradice los falsos positivos de test.** Ultralytics informa la precision en el umbral de confianza que da el mejor resultado, mientras que nosotros predecimos con 0.25.

**Qué haríamos distinto:** más gorras distintas (claras, de colores, colgadas, en exterior), fotos sin gorras con zonas oscuras, separar las particiones **por gorra**, reemplazar el flip vertical por cambios de brillo, y mantener la proporción de las fotos en lugar de estirarlas.

## Reproducir

> **Roboflow en el proyecto original vs. en este repo.** Durante el trabajo práctico usamos [Roboflow](https://roboflow.com/) para anotar las fotos, armar la partición, aplicar la aumentación y exportar el dataset, y el notebook original lo descargaba con una API key. Las fotos de prueba y los pesos entrenados estaban en el Google Drive del grupo.
>
> Para que cualquiera pueda reproducirlo sin cuentas ni permisos, este repo trae todo incluido: el export de Roboflow (versión 2) en `datos/dataset/`, las fotos de prueba en `datos/fotos_prueba/` y los pesos entrenados en `modelos/`. **No hace falta Roboflow, Google Drive ni ninguna API key:** se clona, se instalan las dependencias y se ejecuta.

**Requisitos:** Python 3.11 o superior (probado con 3.13) y `pip`. La GPU es opcional.

```bash
git clone https://github.com/javi2481/detector-gorras-yolo11.git
cd detector-gorras-yolo11
pip install -r requirements.txt
```

Después, abrir `notebooks/TP2_Detector_Gorras.ipynb` (VS Code, Jupyter o Colab) y *Reiniciar y ejecutar todo*.

```mermaid
flowchart TD
    S(["Reiniciar y ejecutar todo"]) --> Q{"ENTRENAR"}
    Q -->|"False (por defecto)"| P["Usa los pesos de modelos/A y modelos/B"]
    Q -->|"True"| T["Reentrena A y B desde cero<br/>sobre datos/dataset"]
    P --> E["Evaluación, test, errores e in the wild"]
    T --> E
    E --> R[("runs/<br/>predicciones y copia de resultados")]

    classDef inicio fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef rapido fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef lento fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef salida fill:#f3f4f6,stroke:#6b7280,color:#1f2937

    class S,Q inicio
    class P,E rapido
    class T lento
    class R salida
```

| Modo | Cómo se activa | Tiempo aproximado | Resultado |
|---|---|---|---|
| Pesos entrenados | `ENTRENAR = False` (por defecto) | Menos de 1 min, incluso en CPU | Las mismas métricas de la entrega |
| Reentrenar | `ENTRENAR = True` en la celda de configuración | ≈ 12 min con GPU · varias horas en CPU | Métricas muy parecidas; en CPU pueden variar levemente |

En Colab, si el notebook no encuentra la carpeta `datos/`, clona este repo automáticamente.

## Estructura del repositorio

```text
.
├── datos/
│   ├── dataset/                     # export de Roboflow v2 (formato YOLOv11)
│   │   ├── data.yaml                # rutas relativas y la clase "Gorra"
│   │   ├── train/                   # images/ + labels/ (180 imágenes aumentadas)
│   │   ├── valid/                   # images/ + labels/ (13 imágenes)
│   │   └── test/                    # images/ + labels/ (6 imágenes)
│   └── fotos_prueba/
│       ├── linea_base/              # 3 fotos para probar yolo11n.pt sin entrenar
│       └── in_the_wild/             # 3 fotos nuevas, fuera de todas las particiones
├── modelos/
│   ├── A/                           # yolo11n (ganador)
│   │   ├── weights/best.pt          # pesos entrenados
│   │   ├── results.csv              # pérdidas y métricas por época
│   │   ├── results.png              # curvas de entrenamiento
│   │   └── val_batch0_pred.jpg      # predicciones sobre un lote de validación
│   └── B/                           # yolo11s: weights/best.pt, results.csv, results.png
├── notebooks/
│   └── TP2_Detector_Gorras.ipynb    # notebook ejecutado, con todas las salidas
├── figuras/                         # imágenes de este README (salen del notebook)
├── resultados/
│   ├── tabla_A_vs_B.csv             # métricas de las dos configuraciones
│   └── casos_test_valid.csv         # cada predicción clasificada (TP / FP / FN / caja mal ajustada)
├── docs/
│   └── consigna.md                  # consigna del trabajo práctico
├── requirements.txt
├── LICENSE                          # licencia MIT
└── README.md
```

Al ejecutar el notebook se crea `runs/` (predicciones de Ultralytics y una copia de los resultados en `runs/entrega/`). Esa carpeta está en `.gitignore`.

Cómo se conectan las carpetas con el notebook:

```mermaid
flowchart LR
    subgraph IN["Entradas (en el repo)"]
        D[("datos/dataset")]
        F[("datos/fotos_prueba")]
        M[("modelos/A · modelos/B")]
    end

    N["notebooks/<br/>TP2_Detector_Gorras.ipynb"]

    subgraph OUT["Salidas"]
        R[("runs/<br/>se genera al ejecutar")]
        RE[("resultados/<br/>tablas CSV")]
        FG[("figuras/<br/>imágenes del README")]
    end

    D --> N
    F --> N
    M --> N
    N --> R
    N -.->|"curadas a mano"| RE
    N -.->|"curadas a mano"| FG

    classDef entrada fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef nb fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef salida fill:#dcfce7,stroke:#16a34a,color:#14532d

    class D,F,M entrada
    class N nb
    class R,RE,FG salida
```

## El notebook, sección por sección

| Sección | Qué hace | Lee | Escribe |
|---|---|---|---|
| 0. Preparación del entorno | Instala dependencias y define rutas, semilla y `ENTRENAR` | — | — |
| 1. Dataset y anotación | Criterio de anotación, partición, estadísticas y ejemplos anotados | `datos/dataset` | `runs/data.yaml` |
| 2. Línea de base | Prueba `yolo11n.pt` sin entrenar sobre 3 fotos | `datos/fotos_prueba/linea_base` | `runs/detect/baseline` |
| 3. Fine-tuning | Configuraciones A y B, curvas y comparación | `modelos/` o `datos/dataset` | `runs/detect/train_*` (solo si `ENTRENAR = True`) |
| 4.1 – 4.3 Evaluación | Métricas sobre valid, inferencia sobre test y análisis de errores | `datos/dataset` | `runs/detect/val_*`, `runs/detect/test_ganador` |
| 4.4 In the wild | Inferencia sobre fotos nuevas, originales y estiradas | `datos/fotos_prueba/in_the_wild` | `runs/detect/in_the_wild` |
| 4.5 Guardado | Copia pesos, tablas y predicciones | `runs/` | `runs/entrega` |
| Reporte final | Problema, resultados, errores y conclusiones | — | — |

## Equipo

| Rol | Integrante | Responsabilidad |
|---|---|---|
| 1 | Mauricio Chocholacek | Roboflow: revisión de recuadros, partición, aumentación y exportación |
| 2 | Paola Lucia Pierolivo Fernandez | Criterio de anotación, estadísticas y línea de base |
| 3 | Veronica Andrea Gonzalez | Configuración A y comparación A vs. B |
| 4 | Daiana Marilyn Ruiz Díaz | Configuración B y fotos "in the wild" |
| 5 | Andrea Romeo | Inferencia sobre test y análisis de errores |
| 6 | Rodolfo Javier Berrone | Notebook final, introducción y conclusiones |

Así se repartió el trabajo a lo largo del proyecto:

```mermaid
flowchart LR
    R1["👤 Mauricio<br/>Rol 1"] --> F1["Dataset en Roboflow<br/>partición · aumentación · export"]
    R2["👤 Paola<br/>Rol 2"] --> F1
    R2 --> F2["Línea de base"]
    R3["👤 Veronica<br/>Rol 3"] --> F3["Configuración A<br/>y comparación A vs. B"]
    R4["👤 Daiana<br/>Rol 4"] --> F4["Configuración B<br/>y fotos in the wild"]
    R5["👤 Andrea<br/>Rol 5"] --> F5["Test y<br/>análisis de errores"]
    R6["👤 Rodolfo<br/>Rol 6"] --> F6["Notebook final<br/>y conclusiones"]

    F1 --> F2 --> F3 --> F5
    F1 --> F4 --> F5
    F5 --> F6

    classDef persona fill:#f3e8ff,stroke:#9333ea,color:#581c87
    classDef fase fill:#dbeafe,stroke:#2563eb,color:#1e3a8a

    class R1,R2,R3,R4,R5,R6 persona
    class F1,F2,F3,F4,F5,F6 fase
```

## Glosario rápido

| Término | Qué significa |
|---|---|
| **Fine-tuning** | Partir de un modelo ya entrenado y reentrenarlo con datos propios |
| **IoU** | Cuánto se superponen dos cajas: 1 si son idénticas, 0 si no se tocan |
| **Precision** | De lo que detectó, cuánto era gorra |
| **Recall** | De las gorras reales, cuántas encontró |
| **mAP@50** | Calidad de detección, contando como acierto una caja con IoU ≥ 0.5 |
| **mAP@50-95** | Lo mismo, pero exigiendo cajas cada vez más ajustadas |

## Créditos

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/): modelos pre-entrenados y entrenamiento.
- [Roboflow](https://roboflow.com/): anotación, partición, aumentación y exportación del dataset.
- Fotos tomadas y anotadas por el equipo.

## Licencia

Proyecto académico distribuido bajo la [licencia MIT](LICENSE).
