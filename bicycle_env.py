"""
Bicicleta de Randløv & Alstrøm (apéndice A) como ambiente Gymnasium,
más un wrapper de sensor con delay, ventana, ruido y máscara.

Convenciones (como el código original del paper):
  - la bici arranca en equilibrio, el neumático trasero en (0, 0) y el
    delantero en (0, l), mirando hacia +y.
  - psi = rumbo (0 = +y). Un psi positivo gira hacia -x.
  - la meta es un círculo de 10 m de radio. Por defecto está a 1000 m a la
    derecha del rumbo inicial (+x), que equivale a "arrancar rumbo oeste y
    meta al norte" del paper.
"""
import math
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- Tabla 1 del paper -------------------------------------------------
C = 0.66          # m, distancia horizontal neumático delantero - CM
D_CM = 0.30       # m, distancia vertical CM bici - CM ciclista
H = 0.94          # m, altura del CM
L = 1.11          # m, distancia entre neumáticos
M_C = 15.0        # kg bici
M_D = 1.7         # kg neumático
M_P = 60.0        # kg ciclista
R = 0.34          # m, radio del neumático
V = 10.0 / 3.6    # m/s, velocidad (10 km/h)
G = 9.82
M = M_C + M_P
SIGMA_DOT = V / R

I_BC = (13.0 / 3.0) * M_C * H**2 + M_P * (H + D_CM) ** 2   # ec. (7)
I_DC = M_D * R**2                                          # ec. (8)
I_DV = 1.5 * M_D * R**2                                    # ec. (9)
I_DL = 0.5 * M_D * R**2                                    # ec. (10)

OMEGA_FALL = math.pi / 15       # 12 grados
THETA_MAX = 1.3963              # 80 grados, tope del manubrio (código original)
GOAL_RADIUS = 10.0


def _sign(x):
    return (x > 0) - (x < 0)


class BicycleEnv(gym.Env):
    """
    Observación (float32): [theta, theta_dot, omega, omega_dot, omega_ddot, psi_goal]
    task = "balance": recompensa -1 si cae, 0 en otro caso.
    task = "goal":    recompensa del paper (-1 caída, +0.01 meta,
                      (4 - psi_g^2) * 0.00004 en el resto).
    action_mode = "discrete": 9 acciones = {T} x {d}, T en {-2,0,2} N,
                              d en {-2,0,2} cm (+ ruido de 2 cm).
    action_mode = "continuous": Box([-1,1]^2) escalado a T en [-2,2], d en [-2,2] cm.
    dt es el paso de simulación; frame_skip repite la acción varios pasos.
    """

    metadata = {"render_modes": []}

    def __init__(self, task="goal", action_mode="discrete", dt=0.01,
                 frame_skip=1, max_seconds=1000.0, goal=(1000.0, 0.0),
                 noise_cm=2.0, reward_scale=0.00004):
        super().__init__()
        self.task = task
        self.action_mode = action_mode
        self.dt = dt
        self.frame_skip = frame_skip
        self.max_steps = int(max_seconds / (dt * frame_skip))
        self.goal = np.array(goal, dtype=np.float64)
        self.noise = noise_cm / 100.0
        self.reward_scale = reward_scale

        n_obs = 6 if task == "goal" else 5
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        if action_mode == "discrete":
            self.action_space = spaces.Discrete(9)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    # ------------------------------------------------------------------
    def _decode(self, action):
        if self.action_mode == "discrete":
            T = (-2.0, 0.0, 2.0)[int(action) // 3]
            d = (-0.02, 0.0, 0.02)[int(action) % 3]
        else:
            a = np.clip(action, -1, 1)
            T, d = 2.0 * float(a[0]), 0.02 * float(a[1])
        d += self.noise * self.np_random.uniform(-1.0, 1.0)
        return T, d

    def _goal_angle(self):
        hx, hy = self.xf - self.xb, self.yf - self.yb
        gx, gy = self.goal[0] - self.xb, self.goal[1] - self.yb
        return math.atan2(hx * gy - hy * gx, hx * gx + hy * gy)

    def _obs(self):
        o = [self.theta, self.theta_dot, self.omega, self.omega_dot, self.omega_ddot]
        if self.task == "goal":
            o.append(self._goal_angle())
        return np.array(o, dtype=np.float32)

    # ------------------------------------------------------------------
    def _physics(self, T, d):
        th = self.theta
        phi = self.omega + math.atan(d / H)                         # ec. (1)
        if abs(th) < 1e-9:
            inv_rf = inv_rb = inv_rcm = 0.0
            cross = 0.0
        else:
            inv_rf = abs(math.sin(th)) / L                          # ec. (4)
            inv_rb = abs(math.tan(th)) / L                          # ec. (5)
            rcm = math.sqrt((L - C) ** 2 + L**2 / math.tan(th) ** 2)  # ec. (6)
            inv_rcm = 1.0 / rcm
            cross = _sign(th)
        # ec. (2)
        self.omega_ddot = (1.0 / I_BC) * (
            M * H * G * math.sin(phi)
            - math.cos(phi) * (I_DC * SIGMA_DOT * self.theta_dot
                               + cross * V**2 * (M_D * R * (inv_rf + inv_rb) + M * H * inv_rcm))
        )
        theta_ddot = (T - I_DV * SIGMA_DOT * self.omega_dot) / I_DL  # ec. (3)

        dt = self.dt
        self.omega += self.omega_dot * dt
        self.omega_dot += self.omega_ddot * dt
        self.theta += self.theta_dot * dt
        self.theta_dot += theta_ddot * dt
        if abs(self.theta) > THETA_MAX:                              # manubrio al tope
            self.theta = _sign(self.theta) * THETA_MAX
            self.theta_dot = 0.0

        # posición de los neumáticos
        psi = self.psi
        tf = max(-1.0, min(1.0, V * dt / (2 * L / max(abs(math.sin(self.theta)), 1e-12))))
        tb = max(-1.0, min(1.0, V * dt / (2 * L / max(abs(math.tan(self.theta)), 1e-12))))
        if abs(self.theta) < 1e-9:
            tf = tb = 0.0
        af = psi + self.theta
        self.xf += V * dt * (-math.sin(af + _sign(af) * math.asin(tf)))
        self.yf += V * dt * math.cos(af + _sign(af) * math.asin(tf))
        self.xb += V * dt * (-math.sin(psi + _sign(psi) * math.asin(tb)))
        self.yb += V * dt * math.cos(psi + _sign(psi) * math.asin(tb))

        # nuevo rumbo a partir de los dos neumáticos
        if self.yf == self.yb:
            self.psi = math.pi / 2 if self.xf < self.xb else -math.pi / 2
        else:
            ang = math.atan((self.xb - self.xf) / (self.yf - self.yb))
            self.psi = math.pi + ang if self.yb > self.yf else ang
        self.psi = (self.psi + math.pi) % (2 * math.pi) - math.pi

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.theta = self.theta_dot = 0.0
        self.omega = self.omega_dot = self.omega_ddot = 0.0
        self.xb, self.yb = 0.0, 0.0
        self.xf, self.yf = 0.0, L
        self.psi = 0.0
        self.t = 0
        self.prev_goal_dist = None
        return self._obs(), {}

    def step(self, action):
        T, d = self._decode(action)
        for _ in range(self.frame_skip):
            self._physics(T, d)
            if abs(self.omega) > OMEGA_FALL:
                break
        self.t += 1

        terminated = truncated = False
        info = {"x": self.xb, "y": self.yb, "xf": self.xf, "yf": self.yf}
        if abs(self.omega) > OMEGA_FALL:
            reward, terminated = -1.0, True
            info["fell"] = True
        elif self.task == "goal" and math.hypot(self.xb - self.goal[0],
                                                self.yb - self.goal[1]) < GOAL_RADIUS:
            reward, terminated = 0.01, True
            info["reached_goal"] = True
        elif self.task == "goal":
            pg = self._goal_angle()
            reward = (4.0 - pg**2) * self.reward_scale
        else:
            reward = 0.0
        if not terminated and self.t >= self.max_steps:
            truncated = True
        return self._obs(), reward, terminated, truncated, info


class SensorWrapper(gym.Wrapper):
    """
    Capa entre el estado verdadero y el agente.
      delay    : el agente ve la observación de hace `delay` pasos.
      window   : cuántas observaciones consecutivas ve (terminan en t - delay).
      noise    : desvío estándar de ruido gaussiano sobre cada variable.
      keep     : índices de variables que se ven (None = todas).
      with_actions: agrega las últimas `window + delay` acciones propias.
    Al inicio del episodio la cola se rellena con la observación inicial.
    """

    def __init__(self, env, delay=0, window=1, noise=0.0, keep=None, with_actions=False):
        super().__init__(env)
        self.delay, self.window, self.noise = delay, window, noise
        self.keep = list(range(env.observation_space.shape[0])) if keep is None else list(keep)
        self.with_actions = with_actions
        self.n_act = 1 if env.action_mode == "discrete" else 2
        n_hist = window + delay
        size = window * len(self.keep) + (n_hist * self.n_act if with_actions else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self._obs_q = deque(maxlen=n_hist)
        self._act_q = deque(maxlen=n_hist)

    def _sense(self):
        q = list(self._obs_q)
        shown = q[: self.window]            # las más viejas: terminan en t - delay
        parts = [o[self.keep] for o in shown]
        if self.with_actions:
            parts += [np.atleast_1d(a).astype(np.float32) for a in self._act_q]
        out = np.concatenate(parts).astype(np.float32)
        if self.noise > 0:
            n = self.window * len(self.keep)
            out[:n] += self.np_random.normal(0, self.noise, n).astype(np.float32)
        return out

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        for _ in range(self._obs_q.maxlen):
            self._obs_q.append(obs)
            self._act_q.append(np.zeros(self.n_act, dtype=np.float32))
        return self._sense(), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._obs_q.append(obs)
        self._act_q.append(np.atleast_1d(action).astype(np.float32))
        return self._sense(), r, term, trunc, info
