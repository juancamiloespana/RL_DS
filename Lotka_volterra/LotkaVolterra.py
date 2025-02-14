# Importar libreria para calcular la media
import matplotlib
#matplotlib.use('TkAgg')  # or 'Qt5Agg' for Qt-based GUI
import matplotlib.pyplot as plt
import numpy as np

class LotkaVolterraModel:
    def __init__(self, alpha, beta, delta, gamma, initial_rabbits=100, initial_wolves=25, dt=0.01, total_time=40000):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.initial_rabbits = initial_rabbits
        self.initial_wolves = initial_wolves
        self.dt = dt
        self.total_time = int(total_time)
        self.L = [0]
        self.R = [self.initial_rabbits]
        self.W = [self.initial_wolves]

    def simulate(self):

        for i in range(self.total_time):
            
            if i == 0:              
                wolves = self.initial_wolves
                rabbits = self.initial_rabbits
            else:
                wolves += (wolves_birth - wolves_death) * self.dt 
                rabbits += (rabbits_birth - rabbits_death)* self.dt 

            wolves_birth = self.alpha * (rabbits * wolves)
            wolves_death = self.beta * wolves

            rabbits_birth = self.delta * rabbits
            rabbits_death = self.gamma * (rabbits * wolves)

            self.L.append(wolves)
            self.R.append(rabbits)
            self.W.append(wolves)

        return np.mean(self.L)
        # return wolves
    
    def plot_simulation(self):
        time_points = np.linspace(0, self.total_time, len(self.R))
        plt.plot(time_points, self.R, label='Rabbits')
        plt.plot(time_points, self.W, label='Wolves')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('Lotka-Volterra Simulation')
        plt.legend()
        plt.show()
    
    def plot_simulation_fase(self):
        time_points = np.linspace(0, self.total_time, len(self.R))
        plt.plot(self.W, self.R, label='Wolves vs Rabbits')
        plt.xlabel('Wolves Population')
        plt.ylabel('Rabbits Population')
        plt.title('Lotka-Volterra Simulation')
        plt.legend()
        plt.show()

# Example usage:
# parameters = [0.002, 0.04, 0.1, 0.0025]
# lotka_volterra_instance = LotkaVolterraModel(*parameters)
# result = lotka_volterra_instance.simulate()
# print(f"Average wolves over time: {result}")

# Grafico
#lotka_volterra_instance.plot_simulation()
#lotka_volterra_instance.plot_simulation_fase()
30*1000