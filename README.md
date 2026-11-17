# Aprender a andar en bicicleta con RL: delay y observabilidad parcial

Proyecto final de Aprendizaje por Refuerzo, tema 5 de la consigna: leer *Learning to Drive a Bicycle using Reinforcement Learning* (Randløv y Alstrøm) y recrear la imagen de la tapa del Sutton y Barto.

> **Estado:** este documento describe lo que vamos a hacer. Las hipótesis no están probadas.

## Objetivo

1. Resolver el problema de la bicicleta del paper: primero aprender a equilibrar y después ir a una meta (*shaping*).
2. Agregar una extensión: **delay en el sensado** y **observabilidad parcial**. Queremos ver cuánto se degrada el aprendizaje cuando lo que ve el agente llega con retraso o es incompleto, y hasta qué punto puede aprender.
3. Comparar memoria armada a mano (MLP con ventana de observaciones) contra memoria aprendida (LSTM).
4. Recrear la imagen de la tapa del libro con rutas simuladas de la bici.

## 1. Validación con Sarsa(λ)

Entrenar Sarsa(λ) tabular como en el paper, sin delay y con ventana 1, para comprobar que nuestra bici se equilibra en el orden de unos miles de intentos. Con ventana o delay la tabla es inviable (con ventana de 3 serían unos 3456³ ≈ 4,1·10¹⁰ estados), así que solo se usa como control.

## 2. Barrido principal con PPO y MLP

### Algoritmo

Usamos **PPO** (stable-baselines3) porque es el único que permite cambiar solo la arquitectura (MLP contra LSTM, con `RecurrentPPO` de sb3-contrib) sin cambiar de algoritmo. SAC y DQN no tienen versión recurrente en SB3.

Empezamos con las **9 acciones discretas** del paper, para poder comparar con él y con el Sarsa de validación. Como variante posterior, se pueden probar acciones continuas.

### Qué varía

Con todas las variables a la vista, una ventana de 1 ya contiene toda la información (la observación trae ω̇ y ω̈), así que barrer la ventana con todo daría poco. Por eso se comparan cuatro variantes de **qué ve el agente**:

| Variante | Qué ve | Qué aporta la comparación |
|---|---|---|
| (a) | Todas las variables, ventana 1 | Referencia con toda la información (vieja por el delay) |
| (b) | Solo θ y ω, ventana 1 | (a) contra (b): cuánto cuesta perder las velocidades |
| (c) | Solo θ y ω, ventana 3 | (b) contra (c): cuánto recupera la ventana de observaciones |
| (d) | (c) más acciones propias pasadas | (c) contra (d): cuánto ayuda saber lo que uno mismo hizo durante el delay |

Con delay, el agente nunca ve el presente: decide con una foto de hace *k* pasos. La variante (b) es ambigua (el mismo ángulo pide acciones opuestas según si la bici cae o se recupera), (c) reconstruye la velocidad comparando fotos y (d) además puede estimar qué pasó en el lapso ciego a partir de sus propios torques.

### Grilla

- **Delays:** {0, 2, 5, 10, 20} pasos, los mismos para las cuatro variantes. Se definen en **milisegundos** en el informe y se convierten a pasos según el paso de decisión.
- **Semillas:** 3 por celda.
- **Total:** 5 delays × 4 variantes × 3 semillas = 60 entrenamientos, cada uno desde cero (no hay *curriculum* entre delays).
- **Hiperparámetros:** misma arquitectura e hiperparámetros en todas las celdas. Se afinan en una o dos celdas de referencia y no en cada una, para que las diferencias se deban a lo que ve el agente y no al ajuste.

### Corrida piloto

Antes del barrido completo: una semilla de (a) y de (c) con unos pocos delays. Si el agente aguanta todos, el rango es demasiado fácil y hay que subirlo; si se cae en todos desde delays chicos, hay que bajarlo. Así se ajusta la grilla sin gastar los 60 entrenamientos.

## 3. Comparación con LSTM

`RecurrentPPO` con una observación por paso (ventana 1), comparado contra las variantes (c) y (d) con los mismos delays. La pregunta es si una memoria aprendida reconstruye tanto como la ventana armada a mano, o más.

## 4. Métricas y resultados

- **Segundos sin caerse** (con tope) y **fracción de episodios que llegan al tope**.
- El reward promedio queda como dato secundario: en la tarea de equilibrar solo vale −1 en las caídas y 0 en el resto, así que informa poco.

Resultados a presentar:
- **Un solo mapa de calor** con las variantes en las filas, los delays en las columnas y los segundos sin caerse (promedio de las semillas) como color. Cada casilla indica el promedio y el desvío, para que no se esconda que una semilla aprendió y otra no.
- **Un gráfico de líneas** con el delay en el eje horizontal y una línea por variante, que muestra el escalón con más claridad.
- Las filas del LSTM se suman al mismo mapa para compararlas con (c) y (d).

## 5. Hipótesis a contrastar

- Menos delay da mejor resultado.
- El efecto del delay probablemente sea un escalón y no una curva suave.
- Con delay 0 las cuatro variantes deberían dar resultados parecidos; las diferencias aparecen al subir el delay, con (b) peor y (d) mejor entre las parciales.
- Más ventana no siempre es mejor: una entrada más grande puede volver el aprendizaje más lento.
- Puede que (d) no ayude mucho si la política se apoya sobre todo en la inclinación actual.

## 6. Extensiones opcionales

- **Error del actuador:** que el torque aplicado difiera un poco del pedido (por ejemplo, un 2%), como eje extra del barrido o como forma de entrenar políticas más robustas.
- **Crítico asimétrico:** el actor ve solo lo que entrega el sensor y el crítico ve el estado verdadero durante el entrenamiento.
- **Acciones continuas** en lugar de las 9 discretas.

## 7. Tarea con meta y shaping

- **Etapa 1:** aprender a equilibrar.
- **Etapa 2:** sumar la recompensa de orientación hacia la meta del paper, manteniéndola pequeña frente a la penalización por caída.
- **Experimento opcional:** comparar contra una recompensa de orientación grande, para ver si se reproduce el problema que describe el paper (el agente aprende a orientarse pero no a equilibrar) y si cambia con más entrenamiento. El paper no lo aclara.

## 8. Recreación de la tapa del libro

La tapa son decenas de curvas finas y translúcidas de colores que salen juntas de un punto y se abren en abanico. Para recrearla: simular muchas corridas desde el mismo punto (con ruido, distintas semillas y metas en un abanico de ángulos) y dibujar la ruta de los neumáticos vista desde arriba, como en las figuras 3, 4, 9 y 10 del paper.

## Pendientes

1. Confirmar con el docente la extensión elegida.
2. Validación con Sarsa(λ).
3. Corrida piloto para ajustar el rango de delays.
4. Script de barrido con PPO (delay × variantes × semillas).
5. Barrido con `RecurrentPPO`.
6. Mapa de calor y curvas.
7. Tarea con meta y recreación de la tapa.
8. Informe.

## Referencias

- J. Randløv y P. Alstrøm, *Learning to Drive a Bicycle using Reinforcement Learning and Shaping*.
- R. Sutton y A. Barto, *Reinforcement Learning: An Introduction* (2.ª edición).