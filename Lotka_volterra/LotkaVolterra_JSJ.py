# Importar libreria para calcular la media
import numpy as np
# Importar los graficos
import matplotlib.pyplot as plt

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

        return np.mean(self.L),  np.mean(self.R)
    
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
P = [0.00242, 0.03012, 0.14904, 0.002] 
P = [0.0025, 0.03  , 0.15  , 0.002  ] ###LBFGS
P=[0.00168234, 0.03860196, 0.14987255, 0.00205495] ###Powell
P = [0.00198,0.04,0.15,0.002] ###powersim GE
P = [0.00242, 0.03012, 0.14904, 0.002] 

lotka_volterra_instance = LotkaVolterraModel(*P)
result = lotka_volterra_instance.simulate()
print(f"Average wolves over time: {result}")

# Grafico
lotka_volterra_instance.plot_simulation()
lotka_volterra_instance.plot_simulation_fase()



40000*0.01