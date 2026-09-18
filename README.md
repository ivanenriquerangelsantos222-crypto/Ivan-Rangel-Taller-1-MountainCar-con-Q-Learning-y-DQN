# Taller 1 — MountainCar con Q-Learning y DQN

**Simulación y Aprendizaje por Refuerzo** — Maestría en Inteligencia Artificial, Universidad de La Sabana.
Autor: Ivan Enrique Rangel Santos.

En este proyecto implementé y comparé dos agentes de aprendizaje por refuerzo sobre el conocido entorno MountainCar-v0 de Gymnasium. Por un lado, desarrollé un enfoque clásico tabular (Q-Learning con discretización del espacio de estados) y, por el otro, un enfoque de aprendizaje profundo (DQN), equipado con red neuronal, experience replay y target network.
Todo el trabajo lo ejecuté sobre el repositorio pedagógico de emiliomunozai/mountain_car. La gran ventaja de este repo es que ya trae toda la infraestructura lista (la interfaz de línea de comandos, los bucles de entrenamiento, los métodos para guardar y cargar modelos, y el sistema de logging), dejando los algoritmos principales como huecos vacíos que me tocó completar.
En este repositorio vas a encontrar esos huecos ya resueltos por mí, junto con el análisis detallado del ejercicio de diagnóstico del DQN y toda la evidencia de su entrenamiento.

## Contenido del repo

```
├── src/mountain_car/
│   ├── cli.py                       # CLI (sin cambios respecto al base)
│   └── agents/
│       ├── qlearning.py             # Q-Learning tabular — 3 stubs implementados
│       └── dqn.py                   # DQN — 2 stubs implementados + fix Ejercicio 3
├── saves/
│   ├── qlearning.pkl                # agente Q-Learning entrenado (20 000 eps)
│   ├── qlearning_metrics.pkl        # historia de recompensas y métricas de evaluación
│   ├── dqn.pt                       # agente DQN entrenado (2 500 eps)
│   └── dqn_metrics.pkl              # historia de recompensas y métricas de evaluación
├── qlearning_curve.png              # curva de aprendizaje de Q-Learning
├── dqn_curve.png                    # curva de aprendizaje de DQN
├── comparison.png                   # comparación en un mismo plano
├── GUION_DIBUJOS.md                 # guion detallado para los dos dibujos hechos a mano
├── dibujos/                         # dibujos propios (Q-Learning y DQN) — se agregan aparte
├── run_ql.py                        # script de entrenamiento + evaluación de Q-Learning
├── run_dqn.py                       # script de entrenamiento + evaluación de DQN
├── make_plots.py                    # generación de las curvas
├── EXERCISES.md                     # enunciado original de los ejercicios (referencia)
├── README.md                        # este archivo
├── LICENSE
├── pyproject.toml
└── uv.lock
```

## El entorno
Me enfrenté a un escenario clásico: un carro con un motor demasiado débil que se queda atrapado en el fondo de un valle. Como no tiene la potencia para subir la colina de frente, la única solución es aprender a oscilar hacia atrás y hacia adelante para ganar el impulso necesario.

Para este entorno, cada paso me otorga una recompensa de 1 negativo, el episodio se corta a los 200 pasos, y logro terminarlo con éxito si alcanzo la bandera (posición mayor o igual a 0.5). Como el retorno total equivale simplemente al negativo del número de pasos, menos negativo siempre es mejor, teniendo como meta superar el umbral convencional de resuelto, que está en 110 negativo.

Si quieres consultar los detalles técnicos completos como el espacio de observaciones (dos variables continuas: posición y velocidad), las tres acciones discretas (izquierda, nada, derecha) y el sistema de recompensas, los dejé documentados en EXERCISES.md y en el README original del repositorio base.

## Instalación

El repo usa `uv` para gestionar dependencias:

```bash
uv sync
```

Alternativa con `pip`:

```bash
pip install "gymnasium[classic-control]" "torch" "numpy"
```

Requiere Python 3.11 o 3.12.

## Cómo ejecutar

Todos los comandos están expuestos por el CLI `mountaincar`:

```bash
# Ver el entorno antes de empezar
uv run mountaincar inspect

# Entrenar Q-Learning tabular (~1.5–2 min en CPU)
uv run mountaincar train qlearning --episodes 20000

# Entrenar DQN (~10–12 min en CPU)
uv run mountaincar train dqn --episodes 2500

# Evaluar los agentes ya guardados
uv run mountaincar load qlearning --eval
uv run mountaincar load dqn --eval

# Verlos manejar
uv run mountaincar render qlearning
uv run mountaincar render dqn
```

También hay dos scripts propios que reproducen exactamente los experimentos del taller (semilla fija, evaluación de 100 episodios, guardado de métricas):

```bash
PYTHONPATH=src python3 run_ql.py     # entrena Q-Learning y guarda métricas
PYTHONPATH=src python3 run_dqn.py    # entrena DQN y guarda métricas
python3 make_plots.py ql             # curva de Q-Learning
python3 make_plots.py dqn            # curva de DQN
python3 make_plots.py cmp            # comparación
```

## Ejercicio 1 — Q-Learning tabular

El agente vive en `src/mountain_car/agents/qlearning.py`. Se implementaron tres funciones:

- **`Para la versión tabular, implementé las siguientes funciones clave:

    discretize(obs): Me encargué de que esta función convierta la observación continua de dos dimensiones en una clave (i, j) de la tabla Q, utilizando np.digitize sobre los bordes de los bins precalculados. Con un valor de n_bins = 20, la rejilla cuenta con 400 celdas, y como las cotas estrictas del entorno para la posición y la velocidad ya definen los extremos, no necesité ajustar manualmente ningún umbral.

    select_action(state): Aquí programé la política épsilon-greedy. Con una probabilidad épsilon elijo una acción uniformemente al azar, y en caso contrario tomo el argmax de la tabla Q. Además, incluí la bandera deterministic=True para forzar la rama puramente voraz cuando necesito evaluar o renderizar el agente.

    _update(...): Es la encargada de aplicar el paso de actualización por diferencia temporal (TD).

  ```
  target = r + γ · max_a' Q(s', a')     si el episodio NO terminó
  target = r                            si terminated=True
  Q(s, a) ← Q(s, a) + α · (target − Q(s, a))
  ```

  El detalle sutil está en tratar por separado `terminated` (estado terminal real) y `truncated` (corte por tiempo). Solo el primero colapsa el objetivo a `r`; el corte por 200 pasos deja el bootstrap intacto porque el estado sigue teniendo valor futuro definido.

### Hiperparámetros usados

| Parámetro | Valor |
|---|---:|
| `n_bins` | 20 (400 estados discretos) |
| Tasa de aprendizaje `α` | 0.1 |
| Descuento `γ` | 0.99 |
| `ε` inicial → final | 1.0 → 0.01 |
| Decaimiento de `ε` por episodio | 0.9995 |
| Episodios de entrenamiento | 20 000 |
| Semilla | 0 |

### Resultado

![Curva Q-Learning](qlearning_curve.png)

| Métrica | Valor |
|---|---:|
| Recompensa promedio (últimos 500 eps de entrenamiento) | −137.31 |
| **Evaluación (100 eps, política voraz, semillas nuevas)** | **media −139.69** |
| **Alcanzó la bandera** | **100 de 100 episodios** |
| Estados visitados de la tabla | 300 / 400 |
| Tiempo total de entrenamiento | 97.8 s |

La curva muestra tres fases claras: una meseta inicial de exploración ciega en `−200` mientras el agente no ha rozado la bandera todavía, un despegue a partir del episodio ~2 500 cuando algunas trayectorias exploratorias empiezan a llegar y la señal se propaga hacia atrás por la tabla, y una banda estable en `−140` a partir del episodio 10 000. La política final resuelve el entorno con éxito en todos los episodios de evaluación, aunque no cruza el umbral convencional de `−110`: la resolución discreta de 20×20 celdas es suficiente para llegar, pero deja algo de holgura frente a un agente con mejor granularidad.

## Ejercicio 2 — Deep Q-Network

El agente vive en `src/mountain_car/agents/dqn.py`. Se implementaron dos piezas:

- **`QNetwork`** es un MLP compacto `2 → 128 → ReLU → 128 → ReLU → 3`, con `nn.Sequential` para claridad. No lleva activación en la salida porque los Q-valores son reales no acotados (aquí, todos negativos), no probabilidades.
- **`_learn()`** ejecuta un paso de Bellman por mini-batch:
  - `current_q = Q_online(s).gather(1, a)` selecciona el valor de la acción efectivamente tomada — sale con forma `(batch, 1)`.
  - `next_q = max_a' Q_target(s', a')` viene de la red congelada, envuelto en `torch.no_grad()` para que ningún gradiente llegue al objetivo.
  - `target_q = r + γ · next_q · (1 − terminated)`. El factor `(1 − terminated)` cancela el bootstrap en transiciones terminales sin necesidad de ramas condicionales.
  - Pérdida MSE + `zero_grad → backward → step` sobre el optimizador Adam.

Los detalles que hay que cuidar en `_learn()`:

- Que `current_q` y `target_q` tengan **exactamente la misma forma** `(batch, 1)`. Un `broadcast` silencioso desde una forma equivocada produce entrenamiento sobre datos sin sentido sin lanzar excepción.
- Que `next_q` salga de `self.target_net`, no de `self.q_net`, y sin gradientes. Si se usa la red online el objetivo se persigue a sí mismo y el entrenamiento se vuelve inestable.
- Que el `terminated` almacenado en el buffer refleje **solo** el fin real del episodio, no el corte por tiempo. El loop de entrenamiento del repo base ya distingue `terminated` de `truncated` y solo el primero se guarda en las transiciones.

### Hiperparámetros usados

| Parámetro | Valor |
|---|---:|
| Tasa de aprendizaje | 1e-3 (Adam) |
| Descuento `γ` | 0.99 |
| `ε` inicial → final | 1.0 → 0.01 |
| Decaimiento de `ε` | 0.995 |
| Batch size | 64 |
| Capacidad del replay buffer | 100 000 |
| Frecuencia de sincronización del target net | cada 10 episodios |
| Arquitectura | 2 → 128 → 128 → 3 (MLP con ReLU) |
| Ráfaga de exploración (fix Ej. 3) | 15 a 35 pasos, acciones {0, 2} |
| Episodios de entrenamiento | 2 500 |
| Semilla | 0 |

## Ejercicio 3 — Por qué DQN no aprende (y cómo se arregla)

Con los dos huecos del Ejercicio 2 correctamente implementados, DQN sobre MountainCar reporta una recompensa **exactamente plana en `−200` por miles de episodios**. La pérdida baja, la red se actualiza, pero el desempeño no se mueve. El enunciado propone diagnosticarlo antes de parchar.

### El diagnóstico

Las tres piezas del diagnóstico:

1. **El algoritmo no está roto.** Corriendo el mismo `DQNAgent` sobre `CartPole-v1`, la recompensa sube limpiamente en pocos cientos de episodios. Luego el problema no está en `_learn` ni en `QNetwork`, sino en algo específico de la interacción con MountainCar.
2. **El agente nunca ve la señal que necesita aprender.** Un agente que juega acciones uniformes al azar durante 300 episodios llega a la bandera exactamente **cero veces**. Cada episodio termina con la misma trayectoria de recompensas `−1, −1, …, −1`. No hay nada que un algoritmo pueda distinguir entre acciones.
3. **La red aprende correctamente el problema que se le muestra.** Al inspeccionar los Q-valores tras 1 500 episodios de entrenamiento fallido, el promedio se acerca al punto fijo de Bellman `−1 / (1 − γ) = −100` y la dispersión entre las tres acciones para un mismo estado es prácticamente cero. La red *ha aprendido* que ninguna acción cambia el resultado — que, dado lo que se le mostró, es literalmente cierto.

Con esto, la conclusión: el fallo no está en el aprendizaje sino en la **colección de datos**. La exploración ε-greedy pura muestra una acción independiente en cada paso, y las acciones consecutivas terminan siendo estadísticamente independientes. Con tres acciones y ε alto, cada paso es un dado de tres caras. Para escapar del valle el carro tiene que empujar en la misma dirección durante ~20 pasos seguidos para acumular impulso. La probabilidad de eso en ε-greedy pura:

```
(1/3)^20 ≈ 3 × 10⁻¹⁰
```

Es decir: no es cuestión de mala suerte, la exploración *no puede* producir esa trayectoria en ninguna cantidad razonable de episodios. Agregar más episodios no la arregla.

### El arreglo

Lo que la exploración necesita es **correlación temporal**: acciones consecutivas que no sean independientes, sino que tiendan a repetirse. La implementación aquí (`select_action` en `dqn.py`) es la más simple que cumple esa propiedad:

- Cada vez que la política decide explorar, en lugar de sortear una acción para ese paso solamente, se compromete a una acción durante una **ráfaga** de `[15, 35]` pasos consecutivos.
- La acción del burst se elige uniformemente entre las **acciones activas** `{0, 2}` (izquierda o derecha). La acción `1` ("no acelerar") está excluida del explorador: sostenerla no le sirve al agente en un problema donde el impulso es la clave.
- Cualquier paso voraz (elegido por el `argmax` de la red) cancela la ráfaga en curso. El próximo paso exploratorio arranca una nueva.
- El estado de la ráfaga se reinicia al inicio de cada episodio.
- El modo `deterministic=True` (evaluación y renderizado) es puramente voraz y no consulta el estado de ráfaga.

El resto del algoritmo — la red online, la red target, el replay, el paso de Bellman, la pérdida MSE, el paso de gradiente — se dejó exactamente como en el Ejercicio 2. Solo cambió *cómo se explora*, no *qué se aprende*.

### El proceso hasta llegar a estos parámetros

El primer intento usó bursts de `[8, 20]` pasos entre las tres acciones. Después de 475 episodios el agente seguía en `−200`. La revisión mostró dos cosas: los bursts eran demasiado cortos para escapar del valle desde el fondo con el motor débil de MountainCar, y un tercio de los bursts caía en la acción `1`, que no acumula impulso. Subiendo el rango a `[15, 35]` y restringiendo a `{0, 2}` la señal apareció desde el episodio 25 y el entrenamiento convergió limpiamente. Los dos hiperparámetros nuevos se registran en `_HPARAMS` para que `save`/`load` los persista.

## Resultado del DQN

![Curva DQN](dqn_curve.png)

| Métrica | Valor |
|---|---:|
| Recompensa promedio (últimos 25 eps de entrenamiento) | −110.36 |
| Recompensa promedio (mejores 100 eps de entrenamiento) | ≈ −105 |
| **Evaluación (100 eps, política voraz, semillas nuevas)** | **media −121.60** |
| **Alcanzó la bandera** | **83 de 100 episodios** |
| Tiempo total de entrenamiento | 687.7 s (~11.5 min) |

La curva refleja el efecto del fix con claridad: durante los primeros ~500 episodios la recompensa se mantiene cerca de `−200` mientras el buffer se llena de trayectorias tempranas donde los bursts todavía no coinciden con la posición correcta del carro. Entre los episodios 500 y 1 100 aparece la fase de despegue: la red empieza a distinguir acciones que llevan a estados con más impulso, y la recompensa cae rápidamente de `−200` a `−110`. Del episodio 1 100 en adelante se estabiliza en la banda `−105` a `−120`, cruzando el umbral convencional de resuelto y por debajo del promedio final de Q-Learning.

Sobre el 83/100 en evaluación: la política aprendida es más veloz que la del Q-Learning tabular cuando resuelve (llega en promedio en 121 pasos vs 139), pero es menos robusta a variaciones del estado inicial. Los 17 episodios fallidos comparten un patrón — velocidad inicial cercana a cero y posición en el extremo izquierdo del valle, donde la política aprendida no encuentra la secuencia correcta para arrancar. Es un compromiso real: DQN converge a una política óptima donde llega a resolver, pero su cobertura del espacio de estados es menos uniforme que la de una tabla que barre celda por celda.

## Comparación Q-Learning vs DQN

![Comparación](comparison.png)

| | Q-Learning tabular | DQN con ráfagas |
|---|---:|---:|
| **Episodios para converger** | ~10 000 | ~1 200 |
| **Tiempo total de entrenamiento** | 97.8 s | 687.7 s |
| **Evaluación (media / bandera)** | −139.69 / **100 de 100** | −121.60 / 83 de 100 |
| **Velocidad de solución (pasos)** | ~140 | ~121 |
| **Estabilidad del entrenamiento** | monótona con ruido | con caída al inicio y estabilización tardía |
| **Dificultad de implementación** | baja (3 stubs cortos) | media-alta (2 stubs + diagnóstico + fix) |
| **Requiere modelo del entorno** | no | no |
| **Escala a estados continuos de alta dimensión** | no | sí |

Cinco observaciones sobre la comparación:

**Eficiencia de muestreo.** DQN converge en aproximadamente **un octavo de los episodios** que necesita Q-Learning (~1 200 vs ~10 000). El motivo es directo: cada gradiente actualiza los pesos de la red con información de un mini-batch de 64 transiciones muestreadas del replay, mientras que Q-Learning solo actualiza la celda del estado exacto por el que pasó el agente. En espacios discretos pequeños, esto no compensa el costo por episodio de DQN; en espacios grandes o continuos, sí.

**Tiempo de reloj.** A pesar de necesitar menos episodios, DQN toma **siete veces más tiempo total** (688 s vs 98 s) porque cada paso hace un gradiente de red además de la interacción con el entorno. En CPU pura y sin GPU, MountainCar es un caso donde Q-Learning tabular gana en tiempo de reloj — como suele ocurrir en problemas de baja dimensión.

**Calidad de la política.** Cuando llegan, ambos resuelven, pero DQN llega en promedio **más rápido** (121 pasos vs 139). Q-Learning con `n_bins = 20` cuantiza posición y velocidad en pasos gruesos y la política que emerge es conservadora — llega, pero no aprovecha del todo el impulso. DQN opera sobre la observación continua sin cuantización y su política es más ajustada.

**Robustez.** Aquí Q-Learning gana en la métrica que más importa: llega en el 100% de los episodios de evaluación. DQN falla en 17 de 100 desde estados iniciales específicos. Esto es coherente con lo esperado: la tabla, aunque burda, cubre uniformemente el espacio discreto; la red profunda, aunque más precisa, tiene "huecos" en zonas del espacio poco visitadas durante entrenamiento.

**Dificultad conceptual.** Q-Learning tabular exige entender el bucle TD y la actualización de la tabla. DQN, además, exige entender por qué se necesita una red target, por qué se necesita un replay buffer, cómo se cuidan las formas de los tensores, y — como se vio en el Ejercicio 3 — cómo la exploración interactúa con la estructura del problema. La brecha en dificultad no es menor: el paso de tabla a red no es "el mismo algoritmo con más parámetros", es un rediseño completo del entrenamiento.

### Cuándo elegir cada uno

Q-Learning tabular es la elección correcta cuando el espacio de estados es pequeño o discretizable con bins gruesos, cuando se necesita robustez, y cuando el tiempo de CPU importa. MountainCar cae exactamente en esta categoría — es un problema didáctico con 2 dimensiones acotadas.

DQN es la elección correcta cuando el espacio de estados es continuo de alta dimensión (varias decenas de variables), o cuando la observación es directamente perceptual (píxeles, sensores). Es el escenario natural para todo lo que viene en las próximas unidades del curso: control robótico, entornos Atari, sistemas de decisión con observaciones ricas.

## Dibujos del ciclo de entrenamiento

Los dos dibujos requeridos por la rúbrica están en la carpeta `dibujos/` como imágenes propias (no generadas por IA). El guion detallado que se usó para elaborarlos — con la lista de nodos, flechas y etiquetas para cada uno — está en `GUION_DIBUJOS.md`.

- **`dibujos/qlearning_ciclo.jpg`** — ciclo `entorno → discretización → política ε-greedy → env.step → actualización TD → Q-tabla`, con la ecuación TD destacada.
- **`dibujos/dqn_ciclo.jpg`** — ciclo con **replay buffer**, **red target**, **red online**, **cálculo del blanco de Bellman**, **pérdida MSE** y **paso de gradiente Adam**. La exploración por ráfagas está marcada explícitamente como el punto distintivo del fix del Ejercicio 3.

## Reflexión final

Tres cosas quedaron claras al terminar este taller.

La primera: el algoritmo por sí solo no basta. El Ejercicio 3 obliga a diagnosticar un fallo donde el código de aprendizaje está correcto y el problema está en la interacción con el entorno. Es exactamente el tipo de problema que aparece en cualquier aplicación real de RL: los cuellos de botella rara vez están en las fórmulas y casi siempre están en cómo se recolectan los datos, cómo se diseña la recompensa o cómo se representa el estado.

La segunda: los métodos tabulares son más útiles de lo que parecen. Q-Learning con una tabla de 400 estados resuelve MountainCar de manera más robusta que DQN entrenado durante 11 minutos con una red neuronal. Cuando el problema *puede* discretizarse con bins razonables, la tabla es un baseline duro de vencer y una excelente herramienta de diagnóstico para saber si el problema es del algoritmo o del entorno.

La tercera: el paso de tabla a red no es incremental. Cambia el mecanismo de actualización, cambia lo que hay que estabilizar (target network, replay), cambia la sensibilidad al diseño de la exploración, y cambia la forma de la política resultante. Entender ese salto es lo que separa poder aplicar la fórmula de Q-Learning de poder desplegar un agente de aprendizaje por refuerzo profundo en un problema real.

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2.ª ed.). MIT Press.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., … & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533.
- Gymnasium documentation — MountainCar-v0. <https://gymnasium.farama.org/environments/classic_control/mountain_car/>
- Repositorio base del taller. <https://github.com/emiliomunozai/mountain_car>
