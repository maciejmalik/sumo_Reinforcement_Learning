class Parameters:
    """Klasa przechowująca wszystkie parametry konfiguracyjne programu"""
    SUMO_BINARY = r"C:\Users\48536\PycharmProjects\pythonProject3\Semstr6\sumo\sumo-win64extra-1.23.1\sumo-1.23.1\bin\sumo.exe"
    godzina_symulacji = 15
    CONFIG_PATH = "godzinowe/traffic_scenario_hour_15.sumocfg"
    PHASES_TO_TUNE = [0, 3, 5, 8]
    STEP_LENGTH = 5
    TRAFFIC_SCALE = 0.4
    DECISION_INTERVAL = 10

    # Parametry sieci neuronowej
    HIDDEN_SIZE = 256
    LEARNING_RATE = 1e-4
    GAMMA = 0.95
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.997
    BATCH_SIZE = 10
    MEMORY_SIZE = 20000
    TARGET_UPDATE_FREQ = 100

    # Dodatkowe parametry
    MAX_PHASE_DURATION = 60
    MIN_PHASE_DURATION = 5
    SIMULATION_DURATION = 3600  # 1 godzina w sekundach
    ACTION_STEP_SIZE = 1  # Krok zmiany czasu fazy

    @classmethod
    def get_state_size(cls):
        """Zwraca rozmiar przestrzeni stanów"""
        return len(cls.PHASES_TO_TUNE)

    @classmethod
    def get_action_size(cls):
        """Zwraca rozmiar przestrzeni akcji"""
        return 3 ** len(cls.PHASES_TO_TUNE)

    @classmethod
    def update_config_path(cls):
        """Aktualizuje ścieżkę konfiguracji na podstawie aktualnej godziny"""
        cls.CONFIG_PATH = f"godzinowe/traffic_scenario_hour_15.sumocfg"