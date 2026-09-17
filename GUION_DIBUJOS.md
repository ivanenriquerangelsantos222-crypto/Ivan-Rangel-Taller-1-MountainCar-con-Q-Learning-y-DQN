# Guion para los dos dibujos del Taller 1

**Instrucciones generales**

- Los dos dibujos deben quedar **a mano** (bolígrafo, tablet, papel), no
  generados por IA — así lo pide la rúbrica explícitamente.
- Formato sugerido: media hoja carta cada uno, orientación horizontal.
- Usa una sola tinta para las estructuras y otra distinta (roja o azul) para
  resaltar la etiqueta del bucle central, para que se vea el ciclo.
- Al final de cada dibujo, deja **una línea de leyenda** con tu nombre y la
  fecha, para que quede claro que es autoría propia.

---

## Dibujo 1 — Ciclo de entrenamiento de Q-Learning tabular

### Qué debe capturar

El ciclo `estado → acción → recompensa → actualización` sobre la Q-tabla, con
la política ε-greedy explícita y la regla de discretización visible.

### Layout sugerido (5 cajas + 1 tabla)

Ordenadas en forma de anillo, con flechas que van en sentido horario:

```
        ┌──────────────┐
        │  ENTORNO     │
        │  MountainCar │
        │   (env)      │
        └──────┬───────┘
               │  obs = (posición, velocidad)
               ▼
        ┌──────────────┐
        │ Discretizar  │     ← "np.digitize a estado (i, j)"
        │  s = (i, j)  │
        └──────┬───────┘
               │  s
               ▼
        ┌────────────────────┐         ┌──────────────────┐
        │ Política ε-greedy  │◄────────│   Q-tabla        │
        │  a = argmax Q(s,·) │         │ Q[s][a] ∈ R^A    │
        │    con prob 1-ε    │  Q(s,·) │   (20×20×3)      │
        │  a = aleatoria     │         └────────▲─────────┘
        │    con prob ε      │                  │
        └────────┬───────────┘                  │
                 │ a                            │  update:
                 ▼                              │  Q[s][a] += α(target - Q[s][a])
        ┌──────────────┐                        │
        │ env.step(a)  │                        │
        │ → s', r,     │                        │
        │   terminated │                        │
        └──────┬───────┘                        │
               │  (s, a, r, s', term)           │
               ▼                                │
        ┌───────────────────────────────────────┴─┐
        │  Actualización TD (Q-Learning)          │
        │  target = r + γ · max_a' Q(s', a')      │
        │  si terminated: target = r              │
        │  Q(s,a) ← Q(s,a) + α · (target − Q(s,a))│
        └─────────────────────────────────────────┘
                 │
                 ▼  (s ← s', repetir hasta done)
             siguiente paso
```

### Elementos que NO deben faltar

Marca cada uno con su etiqueta escrita a mano:

1. **Entorno** — bloque con "MountainCar-v0" y la observación cruda
   `(posición, velocidad)` que devuelve.
2. **Discretización** — flecha etiquetada `np.digitize` que convierte
   la observación continua en un par `(i, j)` de índices de bin. Anota
   al lado: `n_bins = 20 → 400 celdas`.
3. **Q-tabla** — dibújala como una rejilla de 20×20 pequeña, con tres
   valores por celda (uno por acción). Etiqueta: `Q(s, a)`.
4. **Política ε-greedy** — bloque con las dos ramas: `argmax_a Q(s, a)` con
   probabilidad `1 − ε` y `random` con probabilidad `ε`. Marca al lado la
   decadencia: `ε: 1.0 → 0.01, decay 0.9995`.
5. **env.step(a)** — bloque que produce `(s', r, terminated, truncated)`.
6. **Actualización TD** — este es el bloque que debe quedar más grande y
   destacado (usa el color secundario). Escribe la ecuación completa:
   `Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]`
   y anota debajo: `si terminated → target = r`.
7. **Flecha de retroalimentación** desde la actualización hacia la Q-tabla
   (para dejar claro que la tabla se modifica en cada paso).
8. **Bucle** cerrando el ciclo con `s ← s'` hasta que el episodio termine.

### Notas al margen (opcional pero suma)

Escribe a un lado del dibujo, en letra pequeña:

- **Hiperparámetros:** `α = 0.1`, `γ = 0.99`, `n_bins = 20`, `episodios = 20 000`.
- **Terminated vs truncated:** solo `terminated` colapsa el target a `r`;
  el truncado (200 pasos) sigue arrancando con bootstrap.

---

## Dibujo 2 — Ciclo de entrenamiento de DQN

### Qué debe capturar

Los tres ingredientes que distinguen a DQN del Q-Learning tabular:
**replay buffer**, **target network** y **actualización de Bellman por
mini-batch** con SGD.

### Layout sugerido (dos columnas)

Columna izquierda = interacción con el entorno + colección de datos.
Columna derecha = aprendizaje del modelo. Una flecha grande que cruza en
medio conecta las dos.

```
       COLECCIÓN DE DATOS                            APRENDIZAJE

    ┌──────────────┐                            ┌──────────────────┐
    │  ENTORNO     │                            │   Q_target       │
    │  MountainCar │                            │   (red congelada)│
    └──────┬───────┘                            │   copia de Q_θ   │
           │ obs = (posición, velocidad)        │   cada N episodios│
           ▼                                    └──────▲───────────┘
    ┌──────────────────────┐                           │
    │ Política de acción   │◄──── Q_θ(s, ·)     ┌──────┴───────────┐
    │  ε-greedy con        │                    │   Q_online (Q_θ) │
    │  ★EXPLORACIÓN        │                    │   MLP:           │
    │   POR RÁFAGAS★       │                    │   2 → 128 → 128  │
    │  (sostiene la        │                    │       → 3        │
    │   misma acción       │                    └──────▲───────────┘
    │   8-20 pasos)        │                           │
    └────────┬─────────────┘                           │ gradiente
             │ a                                       │ (Adam, lr=1e-3)
             ▼                                         │
    ┌──────────────┐                            ┌──────┴───────────┐
    │ env.step(a)  │                            │  Pérdida MSE     │
    │ → s', r,     │                            │  L = (target_q − │
    │   terminated │                            │       current_q)²│
    └──────┬───────┘                            └──────▲───────────┘
           │ (s, a, r, s', term)                       │
           ▼                                           │
    ┌──────────────────────┐                           │
    │  REPLAY BUFFER       │───── sample(batch=64) ────┤
    │  deque de tamaño 100k│                           │
    │  FIFO de transiciones│                           │
    └──────────────────────┘                           │
                                                       │
                                            ┌──────────┴──────────────┐
                                            │  BLANCO DE BELLMAN      │
                                            │  target_q =             │
                                            │    r + γ · max_a'       │
                                            │      Q_target(s', a')   │
                                            │    · (1 − terminated)   │
                                            │                         │
                                            │  current_q =            │
                                            │    Q_online(s)[a]       │
                                            └─────────────────────────┘
```

### Elementos que NO deben faltar

Marca cada uno con su etiqueta:

1. **Entorno** — igual que en el dibujo 1.
2. **Política ε-greedy con exploración por ráfagas** ⭐ — este es el punto
   distintivo de tu solución al Ejercicio 3. Escribe al lado en letra
   pequeña: *"cuando se decide explorar, la misma acción se mantiene
   durante 8–20 pasos consecutivos — así el auto puede acumular impulso"*.
3. **env.step(a)** — igual que en el dibujo 1.
4. **Replay Buffer** — dibújalo como una fila de casillas apiladas (`deque`)
   con capacidad `100 000`. Muestra que se muestrean **mini-batches de 64**
   transiciones aleatorias hacia el bloque de aprendizaje.
5. **Q_online (Q_θ)** — la red que se entrena. Escribe la arquitectura:
   `2 → 128 → ReLU → 128 → ReLU → 3` (entradas: estado; salidas: Q por
   cada acción).
6. **Q_target** — dibújala como una **copia congelada** de Q_online. Marca
   la flecha desde Q_online que dice "copia cada N=10 episodios" con línea
   punteada.
7. **Bloque de Bellman** — el objeto matemático central. Escribe las dos
   fórmulas:
   - `current_q = Q_online(s)[a]`
   - `target_q = r + γ · max_a' Q_target(s', a') · (1 − terminated)`
8. **Pérdida y gradiente** — `L = MSE(current_q, target_q)`, y una flecha
   que llega a Q_online etiquetada `∇L, Adam, lr=1e-3`. La flecha NO llega
   a Q_target: por eso hay dos redes.
9. **Bucle temporal** — flecha delgada desde `env.step` al entorno para
   dejar claro que la interacción se repite; y otra desde Q_online hacia
   la política, para cerrar el ciclo.

### Notas al margen (opcional pero suma)

- **¿Por qué target network?** Para congelar el objetivo mientras el modelo
  aprende — evita perseguir un blanco móvil.
- **¿Por qué replay buffer?** Para descorrelacionar las muestras y reusar
  experiencia — hace el SGD más eficiente y estable.
- **¿Por qué exploración por ráfagas?** Porque `(1/3)^20 ≈ 3·10⁻¹⁰`:
  la ε-greedy pura de cada paso jamás produce las 20 acciones consecutivas
  que MountainCar exige para escapar del valle.

### Cierre del dibujo

Al pie, en una sola línea, escribe: *"Fix del Ejercicio 3: sustained-run
exploration. El resto del algoritmo (Q_online, Q_target, replay, Bellman)
está intacto."* — deja claro que la corrección fue solo en la exploración,
como pidió el enunciado.
