"""
pso.py
------
Particle Swarm Optimization (PSO) implemented from scratch.

Two PSO variants:
  1. HyperparamPSO — optimizes (lr, batch_size).
  2. ThresholdPSO  — optimizes binarization threshold in [0.1, 0.9].

Adaptive inertia weight (Shi & Eberhart, 1998):
    w(t) = w_max - (w_max - w_min) * (t / T)
"""

import numpy as np
import torch


class HyperparamPSO:

    PARAM_NAMES = ["lr", "batch_size"]

    def __init__(
        self,
        fitness_fn,
        n_particles:   int   = 10,
        n_iters:       int   = 15,
        lr_range:      tuple = (1e-4, 1e-2),
        batch_choices: list  = [8, 16],
        w_max:  float = 0.9,
        w_min:  float = 0.4,
        c1:     float = 1.5,
        c2:     float = 1.5,
        seed:   int   = 42,
    ):
        self.fitness_fn    = fitness_fn
        self.n_particles   = n_particles
        self.n_iters       = n_iters
        self.lr_range      = lr_range
        self.batch_choices = batch_choices
        self.w_max = w_max
        self.w_min = w_min
        self.c1    = c1
        self.c2    = c2
        np.random.seed(seed)

        self.lb = np.array([lr_range[0], 0])
        self.ub = np.array([lr_range[1], len(batch_choices) - 1])

        # global probe counter so every call site can print its index
        self._probe_count = 0
        self._total_probes = n_particles + n_particles * n_iters

    def _decode(self, particle):
        lr     = float(np.clip(particle[0], *self.lr_range))
        bs_idx = int(round(np.clip(particle[1], 0, len(self.batch_choices) - 1)))
        return lr, self.batch_choices[bs_idx]

    def _fitness(self, particle):
        self._probe_count += 1
        lr, batch_size = self._decode(particle)

        # Print BEFORE calling fitness_fn so a hang is immediately visible
        print(f"\n{'='*62}", flush=True)
        print(f"  [Probe {self._probe_count:03d}/{self._total_probes}] "
              f"lr={lr:.5f}  bs={batch_size}", flush=True)
        print(f"{'='*62}", flush=True)

        dice = self.fitness_fn(lr, batch_size)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        print(f"  [Probe {self._probe_count:03d}/{self._total_probes}] "
              f"→ dice={dice:.4f} | "
              f"VRAM={torch.cuda.memory_allocated()/1024**2:.0f}MB",
              flush=True)
        return -dice

    def optimize(self):
        dim = len(self.lb)

        positions  = np.random.uniform(self.lb, self.ub, (self.n_particles, dim))
        velocities = np.zeros_like(positions)

        print(f"\n[PSO-Hyper] Starting swarm initialization")
        print(f"[PSO-Hyper] {self.n_particles} particles × {self.n_iters} iters "
              f"= {self._total_probes} total probe runs", flush=True)

        # ── Swarm initialization ──────────────────────────────────────────
        print(f"\n[PSO-Hyper] Phase 1/2 — initializing all {self.n_particles} particles ...",
              flush=True)
        pbest_pos   = positions.copy()
        pbest_score = np.array([self._fitness(p) for p in positions])

        gbest_idx   = np.argmin(pbest_score)
        gbest_pos   = pbest_pos[gbest_idx].copy()
        gbest_score = pbest_score[gbest_idx]

        print(f"\n[PSO-Hyper] Initialization complete. "
              f"Best so far: dice={-gbest_score:.4f}", flush=True)

        lr0, bs0 = self._decode(gbest_pos)
        for idx, p in enumerate(positions):
            lr, bs = self._decode(p)
            print(f"  Particle {idx:2d}: lr={lr:.5f}, bs={bs} | "
                  f"dice={-pbest_score[idx]:.4f}")

        history = []

        # ── PSO iterations ────────────────────────────────────────────────
        print(f"\n[PSO-Hyper] Phase 2/2 — running {self.n_iters} iterations ...",
              flush=True)

        for t in range(self.n_iters):
            w = self.w_max - (self.w_max - self.w_min) * (t / self.n_iters)
            print(f"\n[PSO-Hyper] ── Iter {t+1:2d}/{self.n_iters} "
                  f"(w={w:.3f}) ──────────────────────", flush=True)

            for i in range(self.n_particles):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * r1 * (pbest_pos[i] - positions[i])
                    + self.c2 * r2 * (gbest_pos    - positions[i])
                )
                positions[i] = np.clip(positions[i] + velocities[i], self.lb, self.ub)

                score = self._fitness(positions[i])

                if score < pbest_score[i]:
                    pbest_score[i] = score
                    pbest_pos[i]   = positions[i].copy()
                    if score < gbest_score:
                        gbest_score = score
                        gbest_pos   = positions[i].copy()

            best_dice = -gbest_score
            history.append(best_dice)
            lr, bs = self._decode(gbest_pos)
            print(f"\n  [Iter {t+1:2d}/{self.n_iters} SUMMARY] "
                  f"w={w:.3f} | best_dice={best_dice:.4f} | "
                  f"lr={lr:.5f}, bs={bs}", flush=True)

        best_lr, best_bs = self._decode(gbest_pos)
        best_params = {"lr": best_lr, "batch_size": best_bs}
        print(f"\n[PSO-Hyper] BEST → {best_params} | Dice={-gbest_score:.4f}\n",
              flush=True)
        return best_params, -gbest_score, history


class ThresholdPSO:

    def __init__(
        self,
        fitness_fn,
        n_particles: int   = 20,
        n_iters:     int   = 30,
        lo:   float  = 0.1,
        hi:   float  = 0.9,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1:   float  = 1.5,
        c2:   float  = 1.5,
        seed: int    = 42,
    ):
        self.fitness_fn  = fitness_fn
        self.n_particles = n_particles
        self.n_iters     = n_iters
        self.lo, self.hi = lo, hi
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        np.random.seed(seed)
        self._eval_count = 0

    def optimize(self):
        positions  = np.random.uniform(self.lo, self.hi, self.n_particles)
        velocities = np.zeros(self.n_particles)

        print(f"\n[PSO-Thresh] {self.n_particles} particles × {self.n_iters} iters",
              flush=True)
        print(f"[PSO-Thresh] Initializing ...", flush=True)

        scores = np.array([-self._eval(p) for p in positions])

        pbest_pos   = positions.copy()
        pbest_score = scores.copy()
        gbest_idx   = np.argmin(pbest_score)
        gbest_pos   = pbest_pos[gbest_idx]
        gbest_score = pbest_score[gbest_idx]

        print(f"[PSO-Thresh] Init done. Best threshold so far: "
              f"{gbest_pos:.4f} → dice={-gbest_score:.4f}", flush=True)

        for t in range(self.n_iters):
            w = self.w_max - (self.w_max - self.w_min) * (t / self.n_iters)

            for i in range(self.n_particles):
                r1 = np.random.rand()
                r2 = np.random.rand()
                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * r1 * (pbest_pos[i] - positions[i])
                    + self.c2 * r2 * (gbest_pos    - positions[i])
                )
                positions[i] = float(np.clip(
                    positions[i] + velocities[i], self.lo, self.hi
                ))
                score = -self._eval(positions[i])
                if score < pbest_score[i]:
                    pbest_score[i] = score
                    pbest_pos[i]   = positions[i]
                    if score < gbest_score:
                        gbest_score = score
                        gbest_pos   = positions[i]

            if (t + 1) % 5 == 0 or t == 0:
                print(f"  [Iter {t+1:2d}/{self.n_iters}] "
                      f"best_threshold={gbest_pos:.4f} | "
                      f"dice={-gbest_score:.4f}", flush=True)

        print(f"\n[PSO-Thresh] BEST threshold={gbest_pos:.4f} | "
              f"Dice={-gbest_score:.4f}\n", flush=True)
        return float(gbest_pos), float(-gbest_score)

    def _eval(self, threshold):
        self._eval_count += 1
        dice = self.fitness_fn(threshold)
        return dice