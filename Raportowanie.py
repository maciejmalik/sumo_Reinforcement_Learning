import matplotlib.pyplot as plt
import numpy as np
from config import Parameters

class ResultPlotter:
    """Klasa odpowiedzialna za wizualizację wyników"""

    @staticmethod
    def plot_results(history):
        """Tworzy i zapisuje wykresy wyników treningu"""
        epochs = [h['epoch'] for h in history]
        penalties = [h['penalty'] for h in history]
        losses = [h['loss'] for h in history]
        epsilons = [h['epsilon'] for h in history]

        plt.figure(figsize=(15, 10))

        # Wykres kary
        plt.subplot(2, 2, 1)
        plt.plot(epochs, penalties, 'b-', alpha=0.5)
        plt.xlabel("Epoka")
        plt.ylabel("Kara (im mniej tym lepiej)")
        plt.title("Kara w czasie")
        plt.grid(True)

        # Średnia krocząca kary
        window = max(1, len(penalties) // 20)
        if window > 1:
            rolling_avg = np.convolve(penalties, np.ones(window) / window, mode='valid')
            plt.plot(epochs[window - 1:], rolling_avg, 'r-', linewidth=2)

        # Wykres strat
        plt.subplot(2, 2, 2)
        valid_losses = [l for l in losses if l is not None and l > 0]
        if valid_losses:
            plt.plot(range(len(valid_losses)), valid_losses, 'g-')
            plt.xlabel("Krok treningowy")
            plt.ylabel("Strata")
            plt.title("Funkcja strat podczas treningu")
            plt.grid(True)

        # Wykres czasów trwania faz
        plt.subplot(2, 2, 3)
        durations = np.array([h['durations'] for h in history])
        for i in range(durations.shape[1]):
            plt.plot(epochs, durations[:, i], label=f"Faza {Parameters.PHASES_TO_TUNE[i]}")
        plt.xlabel("Epoka")
        plt.ylabel("Czas trwania (s)")
        plt.title("Ewolucja czasów trwania faz")
        plt.legend()
        plt.grid(True)

        # Wykres epsilon
        plt.subplot(2, 2, 4)
        plt.plot(epochs, epsilons, 'm-')
        plt.xlabel("Epoka")
        plt.ylabel("Epsilon")
        plt.title("Eksploracja vs eksploatacja")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("wyniki_treningu.png", dpi=300)
        plt.show()