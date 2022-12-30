import random
import numpy as np
import matplotlib.pyplot as plt


class PSO:
    def __init__(self, n_iter_, n_swarm_, n_, b_, d_, w_, c1_, c2_):
        self.iterations_timer = 0
        self.phi = np.linspace(0, 2 * np.pi, 1000)
        self.phi_180 = [self.phi[i] * (180 / np.pi) for i in range(len(self.phi))]

        self.n_iter = n_iter_
        self.n_swarm = n_swarm_
        self.n = n_
        self.b_bounds = b_
        self.d_bounds = d_
        self.w_bounds = w_
        self.bounds = [b_, d_, w_]

        self.c1 = c1_
        self.c2 = c2_

        self.swarm = []
        self.b = []
        self.d = []
        self.w = []

        self.best_values = []

        self.ans = float
        self.Found_Value = float
        self.Fitness_Value = float

        self.best_ans = float
        self.best_found = float
        self.best_fitness = float
        self.best_sll = []
        self.best_bw = []

        self.max_velocity = [b_[0] + b_[1], d_[0] + d_[1], w_[0] + w_[1]]
        self.min_velocity = [-self.max_velocity[0], -self.max_velocity[1], -self.max_velocity[2]]

    class Particles:
        def __init__(self, values_):
            self.Values = values_
            self.BestValues = []
            self.Fitness = 0.0
            self.BestFitness = 0.0
            self.Velocity = []

    def array_factor(self, b_, d_, w_, m_=-60):
        s = 0
        for i in range(self.n):
            psi = (2 * np.pi) * (d_[i]) * (np.cos(self.phi) + b_[i])
            s = s + w_[i] * np.exp(1j * psi * i)
        g = np.abs(s) ** 2
        dbi = 10 * np.log10(g / np.max(g))
        return np.clip(dbi, m_, None)

    def add_value(self):
        self.b = [np.random.uniform(b_bounds[0], b_bounds[1]) for _ in range(self.n)]
        self.d = [np.random.uniform(d_bounds[0], d_bounds[1]) for _ in range(self.n)]
        self.w = [np.random.uniform(w_bounds[0], w_bounds[1]) for _ in range(self.n)]

        return self.b, self.d, self.w

    def update_particles(self):
        r1 = random.random()
        r2 = random.random()

        for x in range(len(self.swarm)):
            for y in range(3):
                for z in range(self.n):
                    self.swarm[x].Velocity[y][z] = self.swarm[x].Velocity[y][z] + \
                                       c1 * r1 * (self.swarm[x].BestValues[y][z] - self.swarm[x].Values[y][z]) + \
                                       c2 * r2 * (self.best_values[y][z] - self.swarm[x].Values[y][z])

                    self.swarm[x].Values[y][z] = self.swarm[x].Values[y][z] + self.swarm[x].Velocity[y][z]

                    if self.swarm[x].Values[y][z] > self.bounds[y][1]:
                        self.swarm[x].Values[y][z] = min(self.swarm[x].Values[y][z], self.bounds[y][1])
                    elif self.swarm[x].Values[y][z] < self.bounds[y][0]:
                        self.swarm[x].Values[y][z] = max(self.swarm[x].Values[y][z], self.bounds[y][0])

    def calculate_fitness(self):
        for Particle in self.swarm:

            self.ans = self.array_factor(Particle.Values[0], Particle.Values[1], Particle.Values[2])

            hp_bw = []
            i = 249
            while i != 0:
                if self.ans[i] < -2.99:
                    hp_bw.append([self.ans[i], (((i + 1) * (2 * np.pi)) / 1000) * (180 / np.pi)])
                    break
                i -= 1

            i = 251
            while i != 500:
                if self.ans[i] < -2.99:
                    hp_bw.append([self.ans[i], (((i + 1) * (2 * np.pi)) / 1000) * (180 / np.pi)])
                    break
                i += 1

            min_dot = 0
            for i in range(251, 500):
                a = self.ans[i]
                b = self.ans[i + 1]
                c = self.ans[i + 2]
                if b <= a and b <= c:
                    if b != a and b != c:
                        min_dot = i + 1
                        break

            sll = [-100, 0]
            for i in range(min_dot, 500):
                if self.ans[i] > sll[0]:
                    sll = [self.ans[i], ((i * (2 * np.pi)) / 1000) * (180 / np.pi)]

            f1 = 0
            for i in range(len(self.ans) - 20):
                f11 = 0
                for j in range(20):
                    f11 = f11 + ((self.phi[j + i + 1] - self.phi[j + i]) * (self.ans[j + i] ** 2))
                f1 = f1 + (f11 / 20)

            f2 = 0
            for i in self.ans:
                f2 = f2 + (i ** 2)

            Particle.Fitness = f1 + f2

            if Particle.Fitness > Particle.BestFitness:
                Particle.BestValues = Particle.Values
                Particle.BestFitness = Particle.Fitness

            if Particle.Fitness > self.Fitness_Value:
                self.best_ans = self.ans
                self.best_values = Particle.Values
                self.Found_Value = Particle.Values
                self.Fitness_Value = Particle.Fitness
                self.best_sll = sll
                self.best_bw = hp_bw

    def main(self):
        self.b = [np.random.uniform(b_bounds[1], b_bounds[1]) for _ in range(self.n)]
        self.d = [np.random.uniform(d_bounds[1], d_bounds[1]) for _ in range(self.n)]
        self.w = [np.random.uniform(w_bounds[1], w_bounds[1]) for _ in range(self.n)]

        self.swarm.append(self.Particles([self.b, self.d, self.w]))

        for i in range(self.n_swarm - 1):
            self.swarm.append(self.Particles(self.add_value()))

        for i in self.swarm:
            i.BestValues = i.Values

        for i in self.swarm:
            i.Velocity = [[random.random() * (self.max_velocity[i] - self.min_velocity[i]) + self.min_velocity[i] for _ in range(self.n)] for i in range(3)]

        self.best_fitness = 0.0
        self.Fitness_Value = 0.0

        while self.iterations_timer != self.n_iter:
            self.calculate_fitness()
            self.swarm = sorted(self.swarm, key=lambda Particles: Particles.Fitness)
            self.update_particles()
            self.iterations_timer += 1
            if self.Fitness_Value > self.best_fitness:
                self.best_fitness = self.Fitness_Value
                self.best_found = self.Found_Value

        print(self.best_sll)

        plt.plot(self.phi_180, self.best_ans, linewidth=2, linestyle='-')
        plt.scatter(self.best_sll[1], self.best_sll[0], linewidths=1, color='red', label="SLL")
        plt.scatter(self.best_bw[0][1], self.best_bw[0][0], linewidths=2, color='orange', label="3db BW")
        plt.scatter(self.best_bw[1][1], self.best_bw[1][0], linewidths=2, color='orange')

        plt.legend()
        font1 = {'family': 'serif', 'color': 'black', 'size': 15}
        plt.title("AF", fontdict=font1)
        plt.xlabel("Theta (degree)")
        plt.ylabel("Array Factor (dB)")
        plt.axis([0, 180, -60, 0])
        plt.grid(axis='y', color='black', linestyle='--', linewidth=0.1)

        plt.show()


n_iter = 20
n_swarm = 20

N = 12
b_bounds = [90.0, 90.0]
d_bounds = [0.5, 0.5]
w_bounds = [0.0, 1.0]

c1 = 2
c2 = 2

Go = PSO(n_iter, n_swarm, N, b_bounds, d_bounds, w_bounds, c1, c2)
Go.main()
