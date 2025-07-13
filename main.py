import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import threading
import time
from datetime import datetime
import subprocess

from config import Parameters
from Uczenie import DNQTrainer
from Sumo import SimulationRunner
from Raportowanie import ResultPlotter

class TrafficLightOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Optymalizator Sygnalizacji Świetlnej")
        self.root.geometry("1200x800")

        # Zmienne dla parametrów
        self.current_hour = tk.IntVar(value=15)
        self.simulation_duration = tk.IntVar(value=3600)
        self.phase_durations = [tk.IntVar(value=25), tk.IntVar(value=25),
                                tk.IntVar(value=35), tk.IntVar(value=25)]

        # Parametry uczenia
        self.epochs = tk.IntVar(value=100)
        self.learning_rate = tk.DoubleVar(value=1e-4)
        self.epsilon_start = tk.DoubleVar(value=1.0)
        self.epsilon_min = tk.DoubleVar(value=0.05)
        self.epsilon_decay = tk.DoubleVar(value=0.997)
        self.batch_size = tk.IntVar(value=10)
        self.hidden_size = tk.IntVar(value=256)
        self.krok = tk.DoubleVar(value=1)


        # Status treningu
        self.training_in_progress = False
        self.simulation_in_progress = False

        # Załaduj zapisane wyniki
        self.saved_results = self.load_all_results()
        # Załaduj bazowe czasy postoju
        self.base_wait_times = self.load_base_wait_times()

        self.create_widgets()
        self.update_config_path()

    def create_widgets(self):
        # Główny notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Zakładka 1: Główne ustawienia
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Główne ustawienia")
        self.create_main_tab(main_frame)

        # Zakładka 2: Parametry uczenia
        learning_frame = ttk.Frame(notebook)
        notebook.add(learning_frame, text="Parametry uczenia")
        self.create_learning_tab(learning_frame)

        # Zakładka 3: Zapisane wyniki
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Zapisane wyniki")
        self.create_results_tab(results_frame)

    def create_main_tab(self, parent):
        # Frame dla ustawień godziny
        hour_frame = ttk.LabelFrame(parent, text="Ustawienia godziny symulacji")
        hour_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(hour_frame, text="Godzina (12-24):").pack(side='left', padx=5)
        hour_scale = ttk.Scale(hour_frame, from_=12, to=24, orient='horizontal',
                               variable=self.current_hour, command=self.on_hour_change)
        hour_scale.pack(side='left', fill='x', expand=True, padx=5)

        self.hour_label = ttk.Label(hour_frame, text="15:00")
        self.hour_label.pack(side='left', padx=5)

        # Frame dla czasów trwania faz
        phases_frame = ttk.LabelFrame(parent, text="Czasy trwania faz świateł (sekundy)")
        phases_frame.pack(fill='x', padx=10, pady=5)

        phase_names = ["Faza 0", "Faza 3", "Faza 5", "Faza 8"]
        for i, (name, var) in enumerate(zip(phase_names, self.phase_durations)):
            frame = ttk.Frame(phases_frame)
            frame.pack(fill='x', padx=5, pady=2)

            ttk.Label(frame, text=f"{name}:", width=10).pack(side='left')
            ttk.Scale(frame, from_=5, to=60, orient='horizontal', variable=var,
                      command=lambda v, i=i: self.on_phase_change(i)).pack(side='left', fill='x', expand=True, padx=5)

            label = ttk.Label(frame, text=f"{var.get()}s", width=5)
            label.pack(side='left')
            setattr(self, f'phase_label_{i}', label)

        # Przycisk resetowania czasów faz
        reset_button = ttk.Button(phases_frame, text="Reset do podstawowych czasów",
                                  command=self.reset_phase_durations)
        reset_button.pack(pady=5)

        # Frame dla czasu symulacji
        sim_frame = ttk.LabelFrame(parent, text="Czas symulacji")
        sim_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(sim_frame, text="Czas symulacji (sekundy):").pack(side='left', padx=5)
        ttk.Scale(sim_frame, from_=300, to=7200, orient='horizontal',
                  variable=self.simulation_duration, command=self.on_duration_change).pack(side='left', fill='x',
                                                                                           expand=True, padx=5)

        self.duration_label = ttk.Label(sim_frame, text="3600s (1h)")
        self.duration_label.pack(side='left', padx=5)

        # Frame dla kroku symulacji
        sim_frame = ttk.LabelFrame(parent, text="Krok symulacji")
        sim_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(sim_frame, text="Krok (sekundy):").pack(side='left', padx=5)
        ttk.Scale(sim_frame, from_=0.1, to=5, orient='horizontal',
                  variable=self.krok, command=self.on_krokss_change).pack(side='left', fill='x',
                                                                                           expand=True, padx=5)

        self.durationss_label = ttk.Label(sim_frame, text="5s")
        self.durationss_label.pack(side='left', padx=5)


        # Przyciski akcji
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(button_frame, text="Uruchom symulację (widoczną)",
                   command=self.run_visible_simulation).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Uruchom symulację (ukrytą)",
                   command=self.run_hidden_simulation).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Rozpocznij uczenie",
                   command=self.start_training).pack(side='left', padx=5)

        # Status
        self.status_var = tk.StringVar(value="Gotowy")
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(status_frame, text="Status:").pack(side='left')
        ttk.Label(status_frame, textvariable=self.status_var).pack(side='left', padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.pack(fill='x', padx=10, pady=5)

    def create_learning_tab(self, parent):
        # Parametry uczenia
        params_frame = ttk.LabelFrame(parent, text="Parametry uczenia")
        params_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Organizacja w dwóch kolumnach
        left_frame = ttk.Frame(params_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)

        right_frame = ttk.Frame(params_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        # Lewa kolumna
        self.create_param_entry(left_frame, "Liczba epok:", self.epochs, 1, 1000)
        self.create_param_entry(left_frame, "Współczynnik uczenia:", self.learning_rate, 0.0001, 0.1, True)
        self.create_param_entry(left_frame, "Epsilon start:", self.epsilon_start, 0.1, 1.0, True)
        self.create_param_entry(left_frame, "Epsilon min:", self.epsilon_min, 0.01, 0.5, True)

        # Prawa kolumna
        self.create_param_entry(right_frame, "Epsilon decay:", self.epsilon_decay, 0.9, 0.999, True)
        self.create_param_entry(right_frame, "Batch size:", self.batch_size, 1, 100)
        self.create_param_entry(right_frame, "Hidden size:", self.hidden_size, 64, 512)

        # Przycisk resetowania do domyślnych
        ttk.Button(params_frame, text="Przywróć domyślne",
                   command=self.reset_learning_params).pack(pady=10)

    def create_param_entry(self, parent, text, variable, min_val, max_val, is_float=False):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)

        ttk.Label(frame, text=text, width=20).pack(side='left')

        if is_float:
            entry = ttk.Entry(frame, textvariable=variable, width=10)
            entry.pack(side='left', padx=5)
        else:
            ttk.Scale(frame, from_=min_val, to=max_val, orient='horizontal',
                      variable=variable).pack(side='left', fill='x', expand=True, padx=5)
            ttk.Label(frame, textvariable=variable, width=8).pack(side='left')

    def create_results_tab(self, parent):
        # Frame dla listy wyników
        list_frame = ttk.LabelFrame(parent, text="Zapisane wyniki uczenia")
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Treeview dla wyników
        columns = ('Godzina', 'Średni czas postoju', 'Poprawa', 'Faza 0', 'Faza 3', 'Faza 5', 'Faza 8', 'Data')
        self.results_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)

        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Przyciski akcji dla wyników
        results_buttons_frame = ttk.Frame(parent)
        results_buttons_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(results_buttons_frame, text="Załaduj wybrany wynik",
                   command=self.load_selected_result).pack(side='left', padx=5)
        ttk.Button(results_buttons_frame, text="Usuń wybrany wynik",
                   command=self.delete_selected_result).pack(side='left', padx=5)
        ttk.Button(results_buttons_frame, text="Eksportuj wyniki",
                   command=self.export_results).pack(side='left', padx=5)

        self.refresh_results_list()

    def on_hour_change(self, value):
        hour = int(float(value))
        self.hour_label.config(text=f"{hour}:00")
        self.update_config_path()

    def on_phase_change(self, phase_index):
        value = int(self.phase_durations[phase_index].get())
        self.phase_durations[phase_index].set(value)  # Zapewnia wartości całkowite
        label = getattr(self, f'phase_label_{phase_index}')
        label.config(text=f"{value}s")

    def on_duration_change(self, value):
        duration = int(float(value))
        self.simulation_duration.set(duration)  # Zapewnia wartości całkowite
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        if hours > 0:
            self.duration_label.config(text=f"{duration}s ({hours}h {minutes}m)")
        else:
            self.duration_label.config(text=f"{duration}s ({minutes}m)")

    def on_krokss_change(self, value):
        duration = float(value)
        self.krok.set(duration)
        self.durationss_label.config(text=f"{duration}")


    def reset_phase_durations(self):
        """Resetuje czasy trwania faz do podstawowych wartości"""
        default_durations = [25, 25, 35, 25]
        for i, duration in enumerate(default_durations):
            self.phase_durations[i].set(duration)
            self.on_phase_change(i)
        messagebox.showinfo("Reset", "Przywrócono podstawowe czasy trwania faz")

    def update_config_path(self):
        hour = int(self.current_hour.get())
        Parameters.godzina_symulacji = hour
        Parameters.CONFIG_PATH = f"godzinowe/traffic_scenario_hour_{hour}.sumocfg"

    def update_parameters(self):
        """Aktualizuje parametry w klasie Parameters"""
        Parameters.SIMULATION_DURATION = int(self.simulation_duration.get())
        Parameters.STEP_LENGTH = self.krok.get()
        Parameters.LEARNING_RATE = self.learning_rate.get()
        Parameters.EPSILON_START = self.epsilon_start.get()
        Parameters.EPSILON_MIN = self.epsilon_min.get()
        Parameters.EPSILON_DECAY = self.epsilon_decay.get()
        Parameters.BATCH_SIZE = int(self.batch_size.get())
        Parameters.HIDDEN_SIZE = int(self.hidden_size.get())

    def get_avg_wait_time_from_penalty(self, penalty):
        """Wyciąga średni czas oczekiwania z kary (zakładając że jest pierwszym składnikiem)"""
        # Kara w SimulationRunner to: avg_wait + 0.1 * max_wait + 10 * teleported + 20 * collissions
        # Dla uproszczenia zakładamy że avg_wait to główny składnik
        return penalty

    def calculate_base_wait_time(self, hour):
        """Oblicza bazowy czas postoju dla danej godziny z domyślnymi parametrami"""
        self.status_var.set(f"Obliczanie bazowego czasu postoju dla godziny {hour}:00...")
        self.progress.start()

        # Ustaw godzinę
        old_hour = Parameters.godzina_symulacji
        Parameters.godzina_symulacji = hour
        Parameters.CONFIG_PATH = f"godzinowe/traffic_scenario_hour_{hour}.sumocfg"

        try:
            default_durations = [25, 25, 35, 25]
            sim_runner = SimulationRunner()
            penalty, _ = sim_runner.run_simulation(default_durations)
            base_wait_time = self.get_avg_wait_time_from_penalty(penalty)

            # Zapisz bazowy czas
            self.base_wait_times[hour] = base_wait_time
            self.save_base_wait_times()

            return base_wait_time
        finally:
            # Przywróć poprzednią godzinę
            Parameters.godzina_symulacji = old_hour
            Parameters.update_config_path()
            self.progress.stop()
            self.status_var.set("Gotowy")

    def load_base_wait_times(self):
        """Ładuje bazowe czasy postoju z pliku"""
        filename = "base_wait_times.json"
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    # Konwertuj klucze z string na int
                    return {int(k): v for k, v in data.items()}
            except Exception as e:
                print(f"Błąd podczas ładowania bazowych czasów: {e}")
        return {}

    def save_base_wait_times(self):
        """Zapisuje bazowe czasy postoju do pliku"""
        filename = "base_wait_times.json"
        try:
            with open(filename, 'w') as f:
                json.dump(self.base_wait_times, f, indent=2)
        except Exception as e:
            print(f"Błąd podczas zapisywania bazowych czasów: {e}")

    def run_visible_simulation(self):
        if self.simulation_in_progress:
            messagebox.showwarning("Uwaga", "Symulacja już trwa!")
            return

        self.simulation_in_progress = True
        self.status_var.set("Uruchamianie widocznej symulacji...")
        self.progress.start()

        def run_sim():
            try:
                # Używamy sumo-gui zamiast sumo
                old_binary = Parameters.SUMO_BINARY
                Parameters.SUMO_BINARY = "C:/Users/48536/PycharmProjects/pythonProject3/Semstr6/sumo/sumo-win64extra-1.23.1/sumo-1.23.1/bin/sumo-gui.exe"

                self.update_parameters()
                durations = [int(var.get()) for var in self.phase_durations]

                sim_runner = SimulationRunner()
                penalty, _ = sim_runner.run_simulation(durations)
                avg_wait_time = self.get_avg_wait_time_from_penalty(penalty)

                Parameters.SUMO_BINARY = old_binary

                self.root.after(0, lambda: self.simulation_finished(avg_wait_time))

            except Exception as e:
                Parameters.SUMO_BINARY = old_binary
                self.root.after(0, lambda: self.simulation_error(str(e)))

        threading.Thread(target=run_sim, daemon=True).start()

    def run_hidden_simulation(self):
        if self.simulation_in_progress:
            messagebox.showwarning("Uwaga", "Symulacja już trwa!")
            return

        self.simulation_in_progress = True
        self.status_var.set("Uruchamianie ukrytej symulacji...")
        self.progress.start()

        def run_sim():
            try:
                self.update_parameters()
                durations = [int(var.get()) for var in self.phase_durations]

                sim_runner = SimulationRunner()
                penalty, _ = sim_runner.run_simulation(durations)
                avg_wait_time = self.get_avg_wait_time_from_penalty(penalty)

                self.root.after(0, lambda: self.simulation_finished(avg_wait_time))

            except Exception as e:
                self.root.after(0, lambda: self.simulation_error(str(e)))

        threading.Thread(target=run_sim, daemon=True).start()

    def simulation_finished(self, avg_wait_time):
        self.simulation_in_progress = False
        self.progress.stop()
        self.status_var.set("Gotowy")
        messagebox.showinfo("Symulacja zakończona", f"Średni czas postoju: {avg_wait_time:.2f}s")

    def simulation_error(self, error_msg):
        self.simulation_in_progress = False
        self.progress.stop()
        self.status_var.set("Błąd symulacji")
        messagebox.showerror("Błąd", f"Błąd podczas symulacji: {error_msg}")

    def start_training(self):
        if self.training_in_progress:
            messagebox.showwarning("Uwaga", "Trening już trwa!")
            return

        hour = int(self.current_hour.get())

        # Sprawdź czy mamy bazowy czas postoju dla tej godziny
        if hour not in self.base_wait_times:
            answer = messagebox.askyesno("Brak bazowego czasu",
                                         f"Nie ma jeszcze bazowego czasu postoju dla godziny {hour}:00.\n"
                                         f"Czy chcesz go teraz obliczyć?")
            if answer:
                try:
                    self.calculate_base_wait_time(hour)
                except Exception as e:
                    messagebox.showerror("Błąd", f"Nie można obliczyć bazowego czasu: {e}")
                    return
            else:
                return

        self.training_in_progress = True
        self.status_var.set("Trening w toku...")
        self.progress.start()

        def train():
            try:
                self.update_parameters()

                trainer = DNQTrainer()
                history = trainer.train_agent(epochs=int(self.epochs.get()))

                # Zapisz wyniki dla aktualnej godziny
                self.save_training_result(trainer, history)

                self.root.after(0, lambda: self.training_finished(trainer, history))

            except Exception as e:
                self.root.after(0, lambda: self.training_error(str(e)))

        threading.Thread(target=train, daemon=True).start()

    def training_finished(self, trainer, history):
        self.training_in_progress = False
        self.progress.stop()
        self.status_var.set("Gotowy")

        # Pokaż wykres
        try:
            plotter = ResultPlotter()
            plotter.plot_results(history)
        except Exception as e:
            print(f"Błąd podczas tworzenia wykresów: {e}")
        # Odśwież listę wyników
        self.saved_results = self.load_all_results()
        self.refresh_results_list()

        # Oblicz poprawę czasu postoju
        hour = int(self.current_hour.get())
        base_wait_time = self.base_wait_times.get(hour, 0)
        best_wait_time = self.get_avg_wait_time_from_penalty(trainer.overall_best_penalty)
        improvement = ((base_wait_time - best_wait_time) / base_wait_time) * 100 if base_wait_time > 0 else 0

        messagebox.showinfo("Trening zakończony",
                            f"Najlepszy średni czas postoju: {best_wait_time:.2f}s\n"
                            f"Poprawa: {improvement:+.1f}%")

    def training_error(self, error_msg):
        self.training_in_progress = False
        self.progress.stop()
        self.status_var.set("Błąd treningu")
        messagebox.showerror("Błąd", f"Błąd podczas treningu: {error_msg}")

    def save_training_result(self, trainer, history):
        """Zapisuje wynik treningu dla konkretnej godziny"""
        hour = int(self.current_hour.get())
        filename = f"results_hour_{hour}.json"

        # Pobierz bazowy czas postoju
        base_wait_time = self.base_wait_times.get(hour, 0)
        best_wait_time = self.get_avg_wait_time_from_penalty(trainer.overall_best_penalty)
        improvement = ((base_wait_time - best_wait_time) / base_wait_time) * 100 if base_wait_time > 0 else 0

        result_data = {
            "hour": hour,
            "best_wait_time": best_wait_time,
            "best_durations": [int(d) for d in trainer.overall_best_durations.tolist()],
            "base_wait_time": base_wait_time,
            "improvement_percent": improvement,
            "training_history": history,
            "training_params": {
                "epochs": int(self.epochs.get()),
                "learning_rate": self.learning_rate.get(),
                "epsilon_start": self.epsilon_start.get(),
                "epsilon_min": self.epsilon_min.get(),
                "epsilon_decay": self.epsilon_decay.get(),
                "batch_size": int(self.batch_size.get()),
                "hidden_size": int(self.hidden_size.get())
            },
            "timestamp": datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)

    def load_all_results(self):
        """Ładuje wszystkie zapisane wyniki"""
        results = {}
        for hour in range(12, 25):
            filename = f"results_hour_{hour}.json"
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        results[hour] = data
                except Exception as e:
                    print(f"Błąd podczas ładowania {filename}: {e}")
        return results

    def refresh_results_list(self):
        """Odświeża listę wyników w interfejsie"""
        # Wyczyść istniejące elementy
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Dodaj wyniki
        for hour, data in sorted(self.saved_results.items()):
            durations = data.get('best_durations', [0, 0, 0, 0])
            improvement = data.get('improvement_percent', 0)
            timestamp = data.get('timestamp', 'N/A')
            if timestamp != 'N/A':
                timestamp = timestamp.split('T')[0]  # Tylko data

            self.results_tree.insert('', 'end', values=(
                f"{hour}:00",
                f"{data.get('best_wait_time', 0):.2f}s",
                f"{improvement:+.1f}%",
                f"{durations[0]}s",
                f"{durations[1]}s",
                f"{durations[2]}s",
                f"{durations[3]}s",
                timestamp
            ))

    def load_selected_result(self):
        """Ładuje wybrany wynik do interfejsu"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("Uwaga", "Wybierz wynik do załadowania!")
            return

        item = self.results_tree.item(selection[0])
        hour_text = item['values'][0]
        hour = int(hour_text.split(':')[0])

        if hour in self.saved_results:
            data = self.saved_results[hour]

            # Ustaw godzinę
            self.current_hour.set(hour)
            self.on_hour_change(hour)

            # Ustaw czasy trwania faz
            durations = data.get('best_durations', [25, 25, 35, 25])
            for i, duration in enumerate(durations):
                if i < len(self.phase_durations):
                    self.phase_durations[i].set(int(duration))
                    self.on_phase_change(i)

            messagebox.showinfo("Sukces", f"Załadowano wynik dla godziny {hour}:00")

    def delete_selected_result(self):
        """Usuwa wybrany wynik"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("Uwaga", "Wybierz wynik do usunięcia!")
            return

        item = self.results_tree.item(selection[0])
        hour_text = item['values'][0]
        hour = int(hour_text.split(':')[0])

        if messagebox.askyesno("Potwierdzenie", f"Czy na pewno usunąć wynik dla godziny {hour}:00?"):
            filename = f"results_hour_{hour}.json"
            if os.path.exists(filename):
                os.remove(filename)

            if hour in self.saved_results:
                del self.saved_results[hour]

            self.refresh_results_list()
            messagebox.showinfo("Sukces", "Wynik został usunięty")

    def export_results(self):
        """Eksportuje wszystkie wyniki do pliku CSV"""
        if not self.saved_results:
            messagebox.showwarning("Uwaga", "Brak wyników do eksportu!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    f.write("Godzina,Średni_czas_postoju,Poprawa_%,Faza_0,Faza_3,Faza_5,Faza_8,Data\n")

                    for hour, data in sorted(self.saved_results.items()):
                        durations = data.get('best_durations', [0, 0, 0, 0])
                        improvement = data.get('improvement_percent', 0)
                        timestamp = data.get('timestamp', 'N/A')
                        if timestamp != 'N/A':
                            timestamp = timestamp.split('T')[0]

                        f.write(f"{hour},{data.get('best_wait_time', 0):.2f},{improvement:.1f},"
                                f"{durations[0]},{durations[1]},{durations[2]},"
                                f"{durations[3]},{timestamp}\n")

                messagebox.showinfo("Sukces", f"Wyniki zostały wyeksportowane do: {filename}")

            except Exception as e:
                messagebox.showerror("Błąd", f"Błąd podczas eksportu: {e}")

    def reset_learning_params(self):
        """Przywraca domyślne parametry uczenia"""
        self.epochs.set(100)
        self.learning_rate.set(1e-4)
        self.epsilon_start.set(1.0)
        self.epsilon_min.set(0.05)
        self.epsilon_decay.set(0.997)
        self.batch_size.set(10)
        self.hidden_size.set(256)
        messagebox.showinfo("Sukces", "Przywrócono domyślne parametry uczenia")


def main():
    root = tk.Tk()
    app = TrafficLightOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()