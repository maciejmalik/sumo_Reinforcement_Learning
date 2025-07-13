import traci
from config import Parameters
from Konwerter import TrafficLightController


class SimulationRunner:
    """Klasa odpowiedzialna za uruchamianie symulacji i zbieranie metryk"""

    def __init__(self):
        self.penalty = 10000.0
        self.traffic_light_id = None

    def run_simulation(self, phase_durations):
        """Uruchamia symulację SUMO i zwraca metryki"""
        try:
            traci.start([
                Parameters.SUMO_BINARY, "-c", Parameters.CONFIG_PATH,
                "--step-length", str(Parameters.STEP_LENGTH),
                "--scale", str(Parameters.TRAFFIC_SCALE),
                "--no-warnings", "true",
                "--time-to-teleport", "-1"
            ])

            # 🔧 Dodano pierwszy krok symulacji
            traci.simulationStep()

            # 🔍 Debug - sprawdź ID sygnalizacji
            traffic_light_id = traci.trafficlight.getIDList()
            print("Traffic light IDs:", traffic_light_id)

            if not traffic_light_id:
                print("Błąd: brak sygnalizacji świetlnej!")
                return self.penalty, phase_durations

            self.traffic_light_id = traffic_light_id[0]
            controller = TrafficLightController(self.traffic_light_id)
            controller.update_phase_durations(phase_durations)

            total_wait = 0
            max_wait = 0
            steps = 0
            current_time = 0

            while current_time < Parameters.SIMULATION_DURATION:
                traci.simulationStep()
                steps += 1
                current_time = traci.simulation.getTime()

                vehicle_ids = traci.vehicle.getIDList()
                for vid in vehicle_ids:
                    wait_time = traci.vehicle.getWaitingTime(vid)
                    if wait_time > 0:
                        total_wait += wait_time
                        if wait_time > max_wait:
                            max_wait = wait_time

            avg_wait = total_wait / max(1, len(vehicle_ids)) if vehicle_ids else 0
            teleported = traci.simulation.getStartingTeleportNumber()
            collisions = traci.simulation.getCollidingVehiclesNumber()

            self.penalty = avg_wait + 0.1 * max_wait + 10 * teleported + 20 * collisions

        except Exception as e:
            print(f"Błąd podczas symulacji: {str(e)}")
            self.penalty = 10000.0
        finally:
            try:
                traci.close()
            except:
                pass

        return self.penalty, phase_durations
