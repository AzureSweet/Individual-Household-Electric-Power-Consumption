# particle_filter.py
import numpy as np
import pandas as pd

class ParticleFilter:
    def __init__(self, n_particles, n_states, hmm_model=None):
        if hmm_model is not None and n_states != hmm_model.n_components:
            raise ValueError(f"n_states ({n_states}) phải khớp với n_components của HMM ({hmm_model.n_components})")
        self.n_particles = n_particles
        self.n_states = n_states
        self.particles = np.random.choice(n_states, size=n_particles)
        self.weights = np.ones(n_particles) / n_particles

    def update(self, observation, hmm_model, scaler):
        if not np.isscalar(observation) or np.isnan(observation):
            raise ValueError("Quan sát phải là một giá trị số hợp lệ.")
        obs_df = pd.DataFrame([[observation]], columns=["Global_active_power"])
        logprob, state_probs = hmm_model.score_samples(scaler.transform(obs_df))
        likelihoods = state_probs[0][self.particles]

        self.weights *= likelihoods
        self.weights += 1e-300  # Tránh chia cho 0
        self.weights /= np.sum(self.weights)

        # Resample nếu cần
        if 1. / np.sum(self.weights ** 2) < self.n_particles / 2:
            indices = np.random.choice(self.n_particles, size=self.n_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles

    def estimate(self):
        return np.bincount(self.particles, weights=self.weights, minlength=self.n_states) / np.sum(self.weights)
    
print("Thành công")