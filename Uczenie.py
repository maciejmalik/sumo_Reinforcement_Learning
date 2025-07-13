import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import traci
import sumolib
import subprocess
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Sumo import SimulationRunner
from config import Parameters

class QNetwork(nn.Module):
    """Sieć neuronowa do aproksymacji funkcji Q"""

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, Parameters.HIDDEN_SIZE)
        self.fc2 = nn.Linear(Parameters.HIDDEN_SIZE, Parameters.HIDDEN_SIZE)
        self.fc3 = nn.Linear(Parameters.HIDDEN_SIZE, Parameters.HIDDEN_SIZE)
        self.out = nn.Linear(Parameters.HIDDEN_SIZE, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class DQNAgent:
    """Agent DQN odpowiedzialny za proces uczenia"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=Parameters.MEMORY_SIZE)
        self.gamma = Parameters.GAMMA
        self.epsilon = Parameters.EPSILON_START
        self.epsilon_decay = Parameters.EPSILON_DECAY
        self.epsilon_min = Parameters.EPSILON_MIN
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Parameters.LEARNING_RATE,
            weight_decay=1e-5
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.steps = 0

    def remember(self, state, action, reward, next_state):
        """Zapamiętuje doświadczenie w pamięci replay"""
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        """Wybieranie akcji na podstawie bieżącego stanu"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        """Proces uczenia na podstawie pamięci replay"""
        if len(self.memory) < Parameters.BATCH_SIZE:
            return None

        minibatch = random.sample(self.memory, Parameters.BATCH_SIZE)
        states, actions, rewards, next_states = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Oblicz wartości Q
        current_qs = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_qs = self.target_model(next_states).detach().max(1)[0]
        target_qs = rewards + self.gamma * next_qs

        # Oblicz stratę
        loss = self.loss_fn(current_qs, target_qs)

        # Optymalizacja
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Aktualizacja epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Aktualizacja modelu docelowego
        self.steps += 1
        if self.steps % Parameters.TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()


class DNQTrainer:
    """Klasa zarządzająca procesem treningowym"""

    def __init__(self):
        self.sim_runner = SimulationRunner()
        self.history = []
        self.overall_best_durations = None
        self.overall_best_penalty = float('inf')

    @staticmethod
    def index_to_action(index, num_phases):
        """Konwertuje indeks akcji na zmiany czasu trwania faz"""
        base = 3
        action = []
        for i in range(num_phases):
            action.append((index // (3 ** i)) % 3 - 1)
        return action

    def train_agent(self, epochs=100):
        """Główna pętla treningowa"""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        state_size = Parameters.get_state_size()
        action_size = Parameters.get_action_size()
        agent = DQNAgent(state_size, action_size)

        initial_durations = np.array([25, 25, 35, 25], dtype=np.float32)
        current_best_durations = initial_durations.copy()
        current_best_penalty = float('inf')

        self.overall_best_durations = current_best_durations.copy()
        self.overall_best_penalty = current_best_penalty

        print("Rozpoczęcie treningu...")
        print(f"Początkowe czasy trwania faz: {initial_durations}")
        print(f"Rozmiar akcji: {action_size}")

        for epoch in range(epochs):
            if epoch % 10 == 0 and epoch > 0:
                perturb = np.random.choice([-10, -5, 5, 10], size=state_size)
                current_best_durations = np.clip(
                    current_best_durations + perturb,
                    Parameters.MIN_PHASE_DURATION,
                    Parameters.MAX_PHASE_DURATION
                )
                print(f"\U0001f500 Losowa perturbacja: {perturb}")

            action_index = agent.act(current_best_durations)
            action = self.index_to_action(action_index, state_size)

            proposed_durations = current_best_durations + np.array(action) * Parameters.ACTION_STEP_SIZE
            new_durations = np.clip(
                proposed_durations,
                Parameters.MIN_PHASE_DURATION,
                Parameters.MAX_PHASE_DURATION
            ).astype(np.float32)

            penalty, _ = self.sim_runner.run_simulation(new_durations)
            if penalty is None:
                penalty = 10000.0
                print(f"Uwaga: symulacja zwróciła None, ustawiono karę na {penalty}")

            reward = -penalty
            agent.remember(current_best_durations.copy(), action_index, reward, new_durations.copy())

            loss = agent.replay()

            improved = False
            if penalty < current_best_penalty:
                current_best_durations = new_durations.copy()
                current_best_penalty = penalty
                improved = True

                if penalty < self.overall_best_penalty:
                    self.overall_best_durations = new_durations.copy()
                    self.overall_best_penalty = penalty
                    print(f"\U0001f525 Nowy rekord! Kara: {penalty:.2f} Czasy: {new_durations}")

            self.history.append({
                "epoch": epoch + 1,
                "durations": new_durations.tolist(),
                "penalty": penalty,
                "reward": reward,
                "epsilon": agent.epsilon,
                "improved": improved,
                "loss": loss if loss is not None else 0.0
            })

            if loss is not None:
                loss_str = f"{loss:.4f}"
            else:
                loss_str = "N/A"

            if (epoch + 1) % 10 == 0 or improved:
                print(
                    f"Epoka {epoch + 1}/{epochs}: Kara: {penalty:.2f} | "
                    f"Czasy: {new_durations} | "
                    f"Epsilon: {agent.epsilon:.3f} | "
                    f"Poprawa: {'TAK' if improved else 'NIE'} | "
                    f"Strata: {loss_str}"
                )

        self.save_results()
        return self.history

    def save_results(self):
        """Zapisuje najlepsze wyniki i historię treningu"""
        with open("best_result.json", "w") as f:
            json.dump({
                "best_durations": self.overall_best_durations.tolist(),
                "best_penalty": self.overall_best_penalty
            }, f, indent=2)

        with open("training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n\U0001f3c6 Trening zakończony! Najlepsza kara: {self.overall_best_penalty:.2f}")
        print(f"Optymalne czasy trwania: {self.overall_best_durations}")