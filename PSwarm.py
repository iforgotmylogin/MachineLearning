import numpy as np
from NeuralNet import NeuralNet
import random

class PSwarm:

    @staticmethod
    def train_network(folds, num_output, label_index, is_classification, Nparticles=20, max_epochs=20, error_tolerance=1e-9, inertia_weight= 0.9, Pbest_influence=1.8, Gbest_influence=1.5):
        # Initialize particles, personal bests, and velocities
        particles = [None] * Nparticles
        PBest = [None] * Nparticles
        PBest_error = [float('inf')] * Nparticles  # Track personal best errors
        velocities = [None] * Nparticles

        # Initialize particles and velocities
        for i in range(Nparticles):
            particles[i] = NeuralNet(folds, 0, 5, num_output)
            initial_weights = particles[i].get_weights()
            velocities[i] = [[random.uniform(-.5, .5) for _ in neuron_weights] for neuron_weights in initial_weights]
            PBest[i] = particles[i].get_weights()

        GBest = particles[0].get_weights()  # Use the first particle's weights as a template
        GBest_error = float('inf')

        epoch = 0
        
        while epoch < max_epochs:
            for i, particle in enumerate(particles):
                # Evaluate the current particle
                error = PSwarm.evaluate_network(particle, folds, label_index, is_classification)

                # Update personal best if current performance is better
                if error < PBest_error[i]:
                    PBest[i] = particle.get_weights()
                    PBest_error[i] = error

                # Update global best if current performance is better than the global best
                if error < GBest_error:
                    GBest = particle.get_weights()
                    GBest_error = error
                    print(f"GBEST {GBest_error}")

                # Update the particle's velocity and position
                new_velocities, new_weights = PSwarm.update_position_and_velocity(
                    PBest[i], GBest, velocities[i], particle.get_weights(),
                    inertia_weight, Pbest_influence, Gbest_influence, epoch, max_epochs
                )
                velocities[i] = new_velocities
                particle.set_weights(new_weights)

            # Check convergence across all particles
            if all(abs(PBest_error[i] - GBest_error) < error_tolerance for i in range(Nparticles)):
                print(f"Converged after {epoch} epochs.")
                break

            epoch += 1

        print(f"Training completed in {epoch} epochs with a final error of {GBest_error:.6f}.")
        # After the training, create a new NeuralNet with the best weights
        best_network = NeuralNet(folds, 2, 5, num_output)  # Make sure the architecture is consistent
        best_network.set_weights(GBest)  # Set the best weights
        return best_network, GBest_error  # Return the NeuralNet instance and error

    @staticmethod
    def update_position_and_velocity(PBest, GBest, velocity, position, inertia_weight, Pbest_influence, Gbest_influence, epoch, max_epochs):
        """
        Update the velocity and position of a particle.
        """
        new_velocity = []
        new_position = []

        # Dynamically adjust the influence parameters based on epoch
        w = inertia_weight * (1 - epoch / max_epochs)  # Decrease inertia over time
        c1 = Pbest_influence * (epoch / max_epochs)  # Increase personal influence over time
        c2 = Gbest_influence * (1 - epoch / max_epochs)  # Decrease global influence over time

        # Iterate over layers
        for layer_idx, layer_weights in enumerate(position):
            layer_velocity = []
            layer_position = []

            # Iterate over neurons in the current layer
            for neuron_idx, weight in enumerate(layer_weights):
                # Calculate the new velocity for the weight with added randomness
                r1 = random.uniform(0, 1)  # Adjusted randomness for exploration
                r2 = random.uniform(0, 1)

                new_vel = (
                    w * velocity[layer_idx][neuron_idx] +  # inertia term
                    c1 * r1 * (PBest[layer_idx][neuron_idx] - weight) +  # cognitive term
                    c2 * r2 * (GBest[layer_idx][neuron_idx] - weight)  # social term
                )
                
                # Append the calculated velocity for the current weight
                layer_velocity.append(new_vel)

                # Update the position (weights) of the particle based on the new velocity
                new_weight = weight + new_vel  # new position is current weight + velocity
                max_weight = 1  # Define as appropriate
                new_weight = np.clip(weight + new_vel, -max_weight, max_weight)
                layer_position.append(new_weight)

            # Append the new layer's velocity and position
            new_velocity.append(layer_velocity)
            new_position.append(layer_position)

        return new_velocity, new_position

    @staticmethod
    def evaluate_network(network, data, label_index, is_classification, lambda_value=0.0000001):
        """
        Evaluate the network's performance and return the error with L2 regularization.
        """
        # Compute the original error
        if is_classification:
            original_error = network.backProp_classification(
                network.feedforwardEpoch(data), label_index, data, epoch=1
            )
        else:
            original_error = network.backProp_regression(
                network.feedforwardEpoch(data), label_index, data, epoch=1
            )

        # Compute L2 regularization term
        weights = network.get_weights()

        # Sum the squared weights across all layers
        l2_regularization = sum(np.sum(layer_weights**2) for layer_weights in weights)
        # Add the regularization term to the original error
        total_error = original_error + (lambda_value * l2_regularization)
        return total_error