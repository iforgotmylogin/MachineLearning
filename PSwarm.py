from NeuralNet import NeuralNet
import random

class PSwarm:
    
    def train_network(folds, num_output, label_index, is_classification):
        Nparticles = 10
        
        global Pbest_influence
        Pbest_influence = 0.8

        global Gbest_influence 
        Gbest_influence = 0.7

        global inertia_weight
        inertia_weight = 0.5

        # Initialize particles, personal bests, and velocities
        particles = [None] * Nparticles
        PBest = [None] * Nparticles
        PBest_error = [float('inf')] * Nparticles  # Track personal best errors
        velocities = [None] * Nparticles

        # Initialize particles and velocities
        for i in range(Nparticles):
            particles[i] = NeuralNet(folds, 2, 5, num_output)
            initial_weights = particles[i].get_weights()
            velocities[i] = [[random.uniform(-1, 1) for _ in neuron_weights] for neuron_weights in initial_weights]
            PBest[i] = particles[i].get_weights()


        GBest = None  # Global best weights
        GBest = particles[0].get_weights()  # Use the first particle's weights as a template
        GBest_error = float('inf')


        epoch = 0
        max_epochs = 250
        error_tolerance = 1e-9
        

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

                # Update the particle's velocity and position
                new_velocities, new_weights = PSwarm.update_position_and_velocity(PBest, GBest, velocities[i], particles[i].get_weights(),inertia_weight, Pbest_influence, Gbest_influence)
                velocities[i] = new_velocities
                particles[i].set_weights(new_weights)

            # Check convergence after all particles are updated
            if abs(GBest_error) < error_tolerance:
                break

            epoch += 1

        print(f"Training completed in {epoch} epochs with a final error of {GBest_error:.6f}.")
        return GBest, GBest_error

    @staticmethod
    def update_position_and_velocity(PBest, GBest, velocity, position, inertia_weight, Pbest_influence, Gbest_influence):
        """
        Update the velocity and position of a particle.
        """
        new_velocity = []
        new_position = []

        # Iterate over layers
        for layer_idx, layer_weights in enumerate(position):
            layer_velocity = []
            layer_position = []

            # Iterate over neurons in the current layer
            for neuron_idx, weight in enumerate(layer_weights):
                # Calculate the new velocity for the weight
                new_vel = (
                    inertia_weight * velocity[layer_idx][neuron_idx] +  # inertia term
                    Pbest_influence * (PBest[layer_idx][neuron_idx] - weight) +  # cognitive term
                    Gbest_influence * (GBest[layer_idx][neuron_idx] - weight)  # social term
                )
                
                # Append the calculated velocity for the current weight
                layer_velocity.append(new_vel)

                # Update the position (weights) of the particle based on the new velocity
                new_weight = weight + new_vel  # new position is current weight + velocity
                layer_position.append(new_weight)

            # Append the new layer's velocity and position
            new_velocity.append(layer_velocity)
            new_position.append(layer_position)

        return new_velocity, new_position
    @staticmethod
    def evaluate_network(network, data, label_index, is_classification):
        """
        Evaluate the network's performance and return the error.
        """
        if is_classification:
            return network.backProp_classification(
                network.feedforwardEpoch(data), label_index, data, epoch=1
            )
        else:
            return network.backProp_regression(
                network.feedforwardEpoch(data), label_index, data, epoch=1
            )
