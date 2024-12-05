from copy import deepcopy
import numpy as np
from NeuralNet import NeuralNet

class PSO:
    def __init__(self, neural_net, data, num_particles=30, num_iterations=100, initial_learning_rate=0.015, decay_rate=0.1):
        self.neural_net = neural_net
        self.data = data
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.num_weights = len(neural_net.get_weights())
        
        # Initialize particles (positions) and velocities
        self.particles = [np.random.uniform(-0.5, 0.5, size=self.num_weights) for _ in range(self.num_particles)]
        self.velocities = [np.random.uniform(-0.5, 0.5, size=self.num_weights) for _ in range(self.num_particles)]
        
        # Initialize personal best positions and scores
        self.personal_best_positions = deepcopy(self.particles)
        self.personal_best_scores = [float('inf')] * self.num_particles
        
        # Initialize global best position
        self.global_best_position = None
        self.global_best_score = float('inf')
        
    def optimize(self):
        for iteration in range(self.num_iterations):
            for i, particle in enumerate(self.particles):
                # Set the neural network weights to the current particle's position
                self.neural_net.set_weights(particle)
                
                # Evaluate the neural network's performance (fitness)
                results = self.neural_net.feedforwardEpoch(self.data)
                fitness = self.neural_net.backProp_classification(results, label_index=-1, data=self.data, 
                                                                  initial_learning_rate=self.initial_learning_rate, 
                                                                  decay_rate=self.decay_rate, epoch=iteration)
                
                # If the current fitness is better than the personal best, update it
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = deepcopy(particle)
                
                # If the current fitness is better than the global best, update it
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = deepcopy(particle)
            
            # Update velocities and positions of particles
            w = 0.5  # Inertia weight
            c1 = 2  # Personal acceleration coefficient
            c2 = 2  # Global acceleration coefficient

            for i in range(self.num_particles):
                # Update velocity
                r1 = np.random.random(self.num_weights)
                r2 = np.random.random(self.num_weights)
                self.velocities[i] = w * self.velocities[i] + c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) + c2 * r2 * (self.global_best_position - self.particles[i])
                
                # Update position
                self.particles[i] = self.particles[i] + self.velocities[i]

        # Return the global best position (best weights found)
        return self.global_best_position