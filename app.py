from statistics import mean
import traceback
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from typing import Sequence

from debug_scenarios import DEBUG_SCENARIO_RUNNERS, run_debug_scenario
from tsp_solver import (
    GAConfig,
    GARunResult,
    GenerationSnapshot,
    TSPInstance,
    bundled_instance_names,
    load_instance_by_name,
    load_tsplib_instance,
    run_ga,
)


class TSPSimulatorUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TSP Simulator (Python)")
        self.root.geometry("1300x780")
        self.root.minsize(1100, 700)
        self.root.configure(bg="#f2f2f2")

        self._init_variables()
        self._build_layout()
        self._wire_state_handlers()
        self._load_selected_instance(initial=True)
        self._reset_visual_state()
        self._set_debug_output(
            "Debug panel ready.\nEnable Debug Mode and run test_1..test_4 for detailed traces."
        )

    def _init_variables(self) -> None:
        instance_names = bundled_instance_names()
        default_instance = instance_names[0] if instance_names else "berlin52.tsp"
        self.instance_var = tk.StringVar(value=default_instance)
        self.show_optimal_path_var = tk.BooleanVar(value=True)
        self.optimal_cost_var = tk.StringVar(value="7542")

        self.population_size_var = tk.IntVar(value=100)
        self.num_generations_var = tk.IntVar(value=150)
        self.elitist_strategy_var = tk.BooleanVar(value=True)
        self.hillclimbing_var = tk.BooleanVar(value=False)
        self.hc_variant_var = tk.StringVar(value="2-opt")
        self.hc_every_n_var = tk.IntVar(value=25)

        self.selection_method_var = tk.StringVar(value="tournament")
        self.tournament_size_var = tk.IntVar(value=3)

        self.crossover_method_var = tk.StringVar(value="ox")
        self.crossover_prob_var = tk.DoubleVar(value=0.80)

        self.mutation_method_var = tk.StringVar(value="2-swap")
        self.mutation_prob_var = tk.DoubleVar(value=0.01)

        self.use_seed_var = tk.BooleanVar(value=True)
        self.seed_value_var = tk.StringVar(value="123")

        self.num_experiments_var = tk.IntVar(value=1)

        self.generation_label_var = tk.StringVar(value="0")
        self.best_tour_label_var = tk.StringVar(value="--")
        self.optimal_tour_label_var = tk.StringVar(value="7542")
        self.status_var = tk.StringVar(value="Ready")

        self.show_best_in_gen_var = tk.BooleanVar(value=True)
        self.show_avg_in_gen_var = tk.BooleanVar(value=True)

        self.debug_enabled_var = tk.BooleanVar(value=True)
        self.debug_test_var = tk.StringVar(value="test_1")

        self.current_instance: TSPInstance | None = None
        self.current_best_route: list[int] | None = None
        self.last_result: GARunResult | None = None
        self.history_best: list[float] = []
        self.history_avg: list[float] = []
        self._canvas_points: list[tuple[float, float]] = []
        self.custom_instances: dict[str, str] = {}

    def _build_layout(self) -> None:
        root_container = ttk.Frame(self.root, padding=10)
        root_container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root_container)
        header.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            header,
            text="TSP Genetic Algorithm Simulator",
            font=("TkDefaultFont", 14, "bold"),
            bg="#f2f2f2",
            fg="#111111",
        ).pack(anchor=tk.W)
        tk.Label(
            header,
            text="Project language: Python",
            bg="#f2f2f2",
            fg="#666666",
        ).pack(anchor=tk.W)

        main = ttk.Frame(root_container)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=0, minsize=360)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

        view = ttk.Frame(main)
        view.grid(row=0, column=1, sticky="nsew")
        view.columnconfigure(0, weight=1)
        view.rowconfigure(0, weight=3, minsize=300)
        view.rowconfigure(2, weight=2, minsize=250)
        view.rowconfigure(3, weight=1, minsize=180)

        self._build_controls_panel(controls)
        self._build_visual_panel(view)
        self._build_footer(root_container)

    def _build_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        self._build_instance_frame(parent).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_ga_parameters_frame(parent).grid(
            row=1, column=0, sticky="ew", pady=(0, 8)
        )
        self._build_operators_frame(parent).grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self._build_run_frame(parent).grid(row=3, column=0, sticky="ew")
        self._build_debug_controls_frame(parent).grid(row=4, column=0, sticky="ew", pady=(8, 0))

    def _build_instance_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="TSP Instance", padding=8)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Read from file").grid(row=0, column=0, sticky="w")
        self.instance_combo = ttk.Combobox(
            frame,
            textvariable=self.instance_var,
            state="readonly",
            values=bundled_instance_names(),
        )
        self.instance_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Button(frame, text="Open .tsp...", command=self._open_custom_instance).grid(
            row=0, column=2, sticky="ew", padx=(8, 0)
        )

        ttk.Checkbutton(
            frame,
            text="Show optimal path",
            variable=self.show_optimal_path_var,
            command=self._draw_instance_view,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 2))

        ttk.Label(frame, text="Optimal cost").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.optimal_cost_var).grid(
            row=2, column=1, sticky="ew", padx=(8, 0)
        )

        return frame

    def _build_ga_parameters_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="GA Parameters", padding=8)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Population size").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            frame,
            from_=2,
            to=100000,
            textvariable=self.population_size_var,
            width=10,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(frame, text="Num of generations").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(
            frame,
            from_=1,
            to=100000,
            textvariable=self.num_generations_var,
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0))

        ttk.Checkbutton(
            frame,
            text="Elitist strategy",
            variable=self.elitist_strategy_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Checkbutton(
            frame,
            text="Hillclimbing (HC)",
            variable=self.hillclimbing_var,
            command=self._toggle_hc_controls,
        ).grid(row=3, column=0, columnspan=2, sticky="w")

        hc_row = ttk.Frame(frame)
        hc_row.grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.hc_2opt_radio = ttk.Radiobutton(
            hc_row,
            text="2-opt",
            value="2-opt",
            variable=self.hc_variant_var,
        )
        self.hc_2opt_radio.pack(side=tk.LEFT)
        self.hc_3opt_radio = ttk.Radiobutton(
            hc_row,
            text="3-opt",
            value="3-opt",
            variable=self.hc_variant_var,
        )
        self.hc_3opt_radio.pack(side=tk.LEFT, padx=(12, 0))

        ttk.Label(frame, text="Generation to start HC").grid(
            row=5, column=0, sticky="w", pady=(4, 0)
        )
        self.hc_every_n_entry = ttk.Entry(frame, textvariable=self.hc_every_n_var, width=10)
        self.hc_every_n_entry.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

        return frame

    def _build_operators_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Operators", padding=8)
        frame.columnconfigure(0, weight=1)

        selection = ttk.LabelFrame(frame, text="Selection", padding=6)
        selection.grid(row=0, column=0, sticky="ew")

        ttk.Radiobutton(
            selection,
            text="Tournament",
            variable=self.selection_method_var,
            value="tournament",
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            selection,
            text="My roulette",
            variable=self.selection_method_var,
            value="my-roulette",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Label(selection, text="Tournament size").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Spinbox(
            selection,
            from_=2,
            to=1000,
            textvariable=self.tournament_size_var,
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(4, 0))

        crossover = ttk.LabelFrame(frame, text="Crossover", padding=6)
        crossover.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Radiobutton(crossover, text="OX", variable=self.crossover_method_var, value="ox").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(crossover, text="CX", variable=self.crossover_method_var, value="cx").grid(
            row=0, column=1, sticky="w", padx=(10, 0)
        )
        ttk.Radiobutton(
            crossover,
            text="PMX",
            variable=self.crossover_method_var,
            value="pmx",
        ).grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Radiobutton(
            crossover,
            text="My crossover",
            variable=self.crossover_method_var,
            value="my-crossover",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(crossover, text="Crossover prob").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(crossover, textvariable=self.crossover_prob_var, width=10).grid(
            row=2, column=1, sticky="w", padx=(10, 0), pady=(4, 0)
        )

        mutation = ttk.LabelFrame(frame, text="Mutation", padding=6)
        mutation.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Radiobutton(
            mutation, text="2-swap", variable=self.mutation_method_var, value="2-swap"
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mutation, text="Shift", variable=self.mutation_method_var, value="shift").grid(
            row=0, column=1, sticky="w", padx=(10, 0)
        )
        ttk.Radiobutton(
            mutation, text="Scramble", variable=self.mutation_method_var, value="scramble"
        ).grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Radiobutton(
            mutation, text="Inversion", variable=self.mutation_method_var, value="inversion"
        ).grid(row=0, column=3, sticky="w", padx=(10, 0))
        ttk.Radiobutton(
            mutation,
            text="My mutation",
            variable=self.mutation_method_var,
            value="my-mutation",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(mutation, text="Mutation prob").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(mutation, textvariable=self.mutation_prob_var, width=10).grid(
            row=2, column=1, sticky="w", padx=(10, 0), pady=(4, 0)
        )

        extras = ttk.Frame(frame)
        extras.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        extras.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            extras,
            text="Use fixed seed",
            variable=self.use_seed_var,
            command=self._toggle_seed_entry,
        ).grid(row=0, column=0, sticky="w")
        self.seed_entry = ttk.Entry(extras, textvariable=self.seed_value_var, width=12)
        self.seed_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(extras, text="Num of experiments").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(
            extras,
            from_=1,
            to=9999,
            textvariable=self.num_experiments_var,
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        return frame

    def _build_run_frame(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ttk.Button(frame, text="RUN", command=self._run_once).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(frame, text="MULTIRUN", command=self._run_multi).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        return frame

    def _build_debug_controls_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Debug / Tests", padding=8)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            frame, text="Debug Mode", variable=self.debug_enabled_var
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(frame, text="Scenario").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.debug_combo = ttk.Combobox(
            frame,
            textvariable=self.debug_test_var,
            state="readonly",
            values=["test_1", "test_2", "test_3", "test_4"],
            width=16,
        )
        self.debug_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        ttk.Button(frame, text="Run Debug Test", command=self._run_debug_scenario).grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(frame, text="Save Debug Log", command=self._save_debug_log).grid(
            row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )

        return frame

    def _build_visual_panel(self, parent: ttk.Frame) -> None:
        instance_frame = ttk.LabelFrame(parent, text="Instance View", padding=6)
        instance_frame.grid(row=0, column=0, sticky="nsew")
        instance_frame.rowconfigure(0, weight=1)
        instance_frame.columnconfigure(0, weight=1)

        self.instance_canvas = tk.Canvas(
            instance_frame,
            background="white",
            highlightthickness=1,
            highlightbackground="#999999",
        )
        self.instance_canvas.grid(row=0, column=0, sticky="nsew")

        stats = ttk.Frame(parent, padding=(0, 8, 0, 8))
        stats.grid(row=1, column=0, sticky="ew")
        for column_idx in (1, 3, 5):
            stats.columnconfigure(column_idx, weight=1)

        ttk.Label(stats, text="Generation:").grid(row=0, column=0, sticky="w")
        ttk.Label(stats, textvariable=self.generation_label_var).grid(row=0, column=1, sticky="w")
        ttk.Label(stats, text="Best tour:").grid(row=0, column=2, sticky="w")
        ttk.Label(stats, textvariable=self.best_tour_label_var).grid(row=0, column=3, sticky="w")
        ttk.Label(stats, text="Optimal tour:").grid(row=0, column=4, sticky="w")
        ttk.Label(stats, textvariable=self.optimal_tour_label_var).grid(row=0, column=5, sticky="w")

        conv_frame = ttk.LabelFrame(parent, text="Convergence", padding=6)
        conv_frame.grid(row=2, column=0, sticky="nsew")
        conv_frame.rowconfigure(0, weight=1)
        conv_frame.columnconfigure(0, weight=1)

        self.convergence_canvas = tk.Canvas(
            conv_frame,
            background="white",
            highlightthickness=1,
            highlightbackground="#999999",
            height=260,
        )
        self.convergence_canvas.grid(row=0, column=0, sticky="nsew")

        legend = ttk.Frame(conv_frame)
        legend.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            legend,
            text="Best in generation",
            variable=self.show_best_in_gen_var,
            command=self._draw_convergence_chart,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            legend,
            text="Average in generation",
            variable=self.show_avg_in_gen_var,
            command=self._draw_convergence_chart,
        ).pack(side=tk.LEFT, padx=(12, 0))

        debug_frame = ttk.LabelFrame(parent, text="Debug Output", padding=6)
        debug_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        debug_frame.rowconfigure(0, weight=1)
        debug_frame.columnconfigure(0, weight=1)

        self.debug_output = scrolledtext.ScrolledText(
            debug_frame,
            wrap=tk.WORD,
            height=10,
            font=("TkFixedFont", 10),
            background="#ffffff",
            foreground="#111111",
            insertbackground="#111111",
            selectbackground="#2f66ff",
            selectforeground="#ffffff",
        )
        self.debug_output.grid(row=0, column=0, sticky="nsew")
        # Read-only behavior without using DISABLED, because some Tk builds
        # do not support disabledforeground for Text/ScrolledText.
        self.debug_output.bind("<Key>", lambda _event: "break")
        self.debug_output.bind("<<Paste>>", lambda _event: "break")
        self.debug_output.bind("<<Cut>>", lambda _event: "break")

    def _build_footer(self, parent: ttk.Frame) -> None:
        footer = ttk.Frame(parent)
        footer.pack(fill=tk.X, pady=(8, 0))
        footer.columnconfigure(0, weight=1)

        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        tk.Label(
            footer,
            text="Credits: Artur Gubanovich, Jahor Falkouski",
            fg="#888888",
            bg="#f2f2f2",
            font=("TkDefaultFont", 8),
        ).grid(row=0, column=1, sticky="e")

    def _wire_state_handlers(self) -> None:
        self.instance_canvas.bind("<Configure>", lambda _: self._draw_instance_view())
        self.convergence_canvas.bind("<Configure>", lambda _: self._draw_convergence_chart())
        self.instance_combo.bind("<<ComboboxSelected>>", lambda _: self._load_selected_instance())
        self._toggle_hc_controls()
        self._toggle_seed_entry()

    def _toggle_hc_controls(self) -> None:
        state = tk.NORMAL if self.hillclimbing_var.get() else tk.DISABLED
        self.hc_2opt_radio.configure(state=state)
        self.hc_3opt_radio.configure(state=state)
        self.hc_every_n_entry.configure(state=state)

    def _toggle_seed_entry(self) -> None:
        state = tk.NORMAL if self.use_seed_var.get() else tk.DISABLED
        self.seed_entry.configure(state=state)

    def _set_debug_output(self, text: str) -> None:
        self.debug_output.delete("1.0", tk.END)
        self.debug_output.insert(tk.END, text.strip() + "\n")
        self.debug_output.see(tk.END)

    def _run_debug_scenario(self) -> None:
        if not self.debug_enabled_var.get():
            self.status_var.set("Debug Mode is disabled. Enable it to run tests.")
            return

        scenario_id = self.debug_test_var.get().strip()
        if scenario_id not in DEBUG_SCENARIO_RUNNERS:
            self.status_var.set(f"Unknown debug scenario: {scenario_id}")
            return

        try:
            result = run_debug_scenario(scenario_id)
            self._set_debug_output(result.as_text())
            status = "PASS" if result.passed else "FAIL"
            self.status_var.set(f"Debug {scenario_id} finished: {status}")
        except Exception as exc:
            self._set_debug_output(f"Debug execution error:\n{exc}")
            self.status_var.set(f"Debug {scenario_id} failed with error.")

    def _save_debug_log(self) -> None:
        content = self.debug_output.get("1.0", tk.END).strip()
        if not content:
            self.status_var.set("Debug output is empty. Run a debug test first.")
            return

        filename = f"{self.debug_test_var.get()}_log.txt"
        path = filedialog.asksaveasfilename(
            title="Save debug log",
            defaultextension=".txt",
            initialfile=filename,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            self.status_var.set("Save canceled.")
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        self.status_var.set(f"Debug log saved: {path}")

    def _open_custom_instance(self) -> None:
        path = filedialog.askopenfilename(
            title="Open TSPLIB instance",
            filetypes=[("TSP files", "*.tsp"), ("All files", "*.*")],
        )
        if not path:
            self.status_var.set("Open instance canceled.")
            return
        try:
            self.current_instance = load_tsplib_instance(path)
            display_name = f"custom:{self.current_instance.name}"
            self.custom_instances[display_name] = path
            current_values = list(self.instance_combo.cget("values"))
            if display_name not in current_values:
                current_values.append(display_name)
                self.instance_combo.configure(values=current_values)
            self.instance_var.set(display_name)
            if self.current_instance.optimal_cost is not None:
                self.optimal_cost_var.set(f"{self.current_instance.optimal_cost:.0f}")
            else:
                self.optimal_cost_var.set("")
            self._reset_visual_state()
            self.status_var.set(
                f"Loaded instance from file: {self.current_instance.name} ({self.current_instance.dimension} cities)."
            )
        except Exception as exc:
            self.status_var.set(f"Failed to load instance: {exc}")

    def _load_selected_instance(self, initial: bool = False) -> None:
        selected = self.instance_var.get().strip()
        try:
            if selected in self.custom_instances:
                self.current_instance = load_tsplib_instance(self.custom_instances[selected])
            else:
                self.current_instance = load_instance_by_name(selected)
        except Exception as exc:
            self.status_var.set(f"Could not load instance '{selected}': {exc}")
            return

        if self.current_instance.optimal_cost is not None:
            self.optimal_cost_var.set(f"{self.current_instance.optimal_cost:.0f}")
            self.optimal_tour_label_var.set(f"{self.current_instance.optimal_cost:.2f}")
        else:
            self.optimal_cost_var.set("")
            self.optimal_tour_label_var.set("--")
        self._reset_visual_state()
        if not initial:
            self.status_var.set(
                f"Instance loaded: {self.current_instance.name} ({self.current_instance.dimension} cities)."
            )

    def _reset_visual_state(self) -> None:
        self.history_best = []
        self.history_avg = []
        self.last_result = None
        self.current_best_route = None
        self.generation_label_var.set("0")
        self.best_tour_label_var.set("--")
        self._draw_instance_view()
        self._draw_convergence_chart()

    def _build_ga_config_from_ui(self) -> GAConfig:
        if self.current_instance is None:
            raise ValueError("No instance loaded")

        try:
            population_size = int(self.population_size_var.get())
            num_generations = int(self.num_generations_var.get())
            tournament_size = int(self.tournament_size_var.get())
            crossover_prob = float(self.crossover_prob_var.get())
            mutation_prob = float(self.mutation_prob_var.get())
            hillclimbing_start_generation = int(self.hc_every_n_var.get())
        except (ValueError, tk.TclError) as exc:
            raise ValueError("Numeric GA parameters are invalid") from exc

        seed: int | None = None
        if self.use_seed_var.get():
            try:
                seed = int(self.seed_value_var.get())
            except (ValueError, tk.TclError) as exc:
                raise ValueError("Seed must be an integer") from exc

        return GAConfig(
            population_size=population_size,
            num_generations=num_generations,
            selection_method=self.selection_method_var.get(),
            tournament_size=tournament_size,
            crossover_method=self.crossover_method_var.get(),
            crossover_prob=crossover_prob,
            mutation_method=self.mutation_method_var.get(),
            mutation_prob=mutation_prob,
            elitist=self.elitist_strategy_var.get(),
            hillclimbing=self.hillclimbing_var.get(),
            hillclimbing_variant=self.hc_variant_var.get(),
            hillclimbing_start_generation=hillclimbing_start_generation,
            seed=seed,
        )

    def _read_optimal_cost_input(self) -> float | None:
        text = self.optimal_cost_var.get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _on_generation_snapshot(self, snapshot: GenerationSnapshot) -> None:
        self.generation_label_var.set(str(snapshot.generation))
        self.best_tour_label_var.set(f"{snapshot.best_cost:.2f}")
        self.current_best_route = list(snapshot.best_route)
        self.history_best.append(snapshot.best_cost)
        self.history_avg.append(snapshot.avg_cost)

        # Keep the UI responsive during long GA loops.
        if snapshot.generation % 5 == 0 or snapshot.generation == 0:
            self._draw_instance_view()
            self._draw_convergence_chart()
            self.root.update_idletasks()

    def _draw_instance_view(self) -> None:
        canvas = self.instance_canvas
        canvas.delete("all")

        if self.current_instance is None:
            return
        cities = self.current_instance.cities
        if len(cities) < 2:
            return

        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 200)
        margin = 20
        usable_w = max(width - 2 * margin, 20)
        usable_h = max(height - 2 * margin, 20)

        xs = [point[0] for point in cities]
        ys = [point[1] for point in cities]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max(max_x - min_x, 1e-9)
        range_y = max(max_y - min_y, 1e-9)

        scale = min(usable_w / range_x, usable_h / range_y)
        offset_x = margin + (usable_w - range_x * scale) / 2.0
        offset_y = margin + (usable_h - range_y * scale) / 2.0

        self._canvas_points = [
            (
                offset_x + (x - min_x) * scale,
                offset_y + (y - min_y) * scale,
            )
            for x, y in cities
        ]

        if self.show_optimal_path_var.get() and self.current_instance.reference_route:
            self._draw_route(self.current_instance.reference_route, color="#8dc5bf", width=1)

        if self.current_best_route:
            self._draw_route(self.current_best_route, color="#2f66ff", width=2)

        for px, py in self._canvas_points:
            canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill="#1f9d97", outline="")

    def _draw_route(self, route: Sequence[int], color: str, width: int) -> None:
        if not route or len(route) < 2:
            return
        for idx in range(len(route)):
            a = route[idx]
            b = route[(idx + 1) % len(route)]
            x1, y1 = self._canvas_points[a]
            x2, y2 = self._canvas_points[b]
            self.instance_canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def _draw_convergence_chart(self) -> None:
        canvas = self.convergence_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 260)
        height = max(canvas.winfo_height(), 180)
        left = 42
        bottom = height - 28
        right = width - 18
        top = 18

        canvas.create_line(left, bottom, right, bottom, fill="#666666")
        canvas.create_line(left, bottom, left, top, fill="#666666")
        canvas.create_text((left + right) / 2, bottom + 16, text="Generation")
        canvas.create_text(18, (top + bottom) / 2, text="Tour\nlength")

        if not self.history_best and not self.history_avg:
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="Run GA to see convergence",
                fill="#999999",
            )
            return

        values: list[float] = []
        if self.show_best_in_gen_var.get() and self.history_best:
            values.extend(self.history_best)
        if self.show_avg_in_gen_var.get() and self.history_avg:
            values.extend(self.history_avg)
        if not values:
            values = self.history_best or self.history_avg

        min_y = min(values)
        max_y = max(values)
        span = max(max_y - min_y, 1e-9)

        max_len = max(len(self.history_best), len(self.history_avg))
        if max_len < 2:
            max_len = 2

        def to_canvas_points(series: Sequence[float]) -> list[float]:
            if len(series) == 1:
                x = left
                y = bottom - (series[0] - min_y) / span * (bottom - top)
                return [x, y]
            points: list[float] = []
            for i, value in enumerate(series):
                x = left + (right - left) * i / (max_len - 1)
                y = bottom - (value - min_y) / span * (bottom - top)
                points.extend((x, y))
            return points

        if self.show_best_in_gen_var.get() and self.history_best:
            best_points = to_canvas_points(self.history_best)
            if len(best_points) >= 4:
                canvas.create_line(*best_points, fill="#2f66ff", width=2, smooth=True)
            else:
                canvas.create_oval(
                    best_points[0] - 2,
                    best_points[1] - 2,
                    best_points[0] + 2,
                    best_points[1] + 2,
                    fill="#2f66ff",
                    outline="",
                )

        if self.show_avg_in_gen_var.get() and self.history_avg:
            avg_points = to_canvas_points(self.history_avg)
            if len(avg_points) >= 4:
                canvas.create_line(*avg_points, fill="#d14a75", width=2, smooth=True)
            else:
                canvas.create_oval(
                    avg_points[0] - 2,
                    avg_points[1] - 2,
                    avg_points[0] + 2,
                    avg_points[1] + 2,
                    fill="#d14a75",
                    outline="",
                )

    def _run_once(self) -> None:
        if self.current_instance is None:
            self.status_var.set("No instance loaded.")
            return
        try:
            config = self._build_ga_config_from_ui()
        except ValueError as exc:
            self.status_var.set(f"Invalid parameters: {exc}")
            return

        self.status_var.set("Running GA...")
        self.root.update_idletasks()

        self.history_best = []
        self.history_avg = []
        self.current_best_route = None

        try:
            self.last_result = run_ga(
                instance=self.current_instance,
                config=config,
                progress_callback=self._on_generation_snapshot,
            )
        except Exception as exc:
            self.status_var.set(f"GA run failed: {exc}")
            return

        self.current_best_route = list(self.last_result.best_route)
        self.best_tour_label_var.set(f"{self.last_result.best_cost:.2f}")
        self.generation_label_var.set(str(config.num_generations))
        optimal_cost = self._read_optimal_cost_input()
        self.optimal_tour_label_var.set(f"{optimal_cost:.2f}" if optimal_cost is not None else "--")
        self._draw_instance_view()
        self._draw_convergence_chart()
        if optimal_cost is not None and optimal_cost > 0:
            gap = ((self.last_result.best_cost - optimal_cost) / optimal_cost) * 100.0
            self.status_var.set(
                f"RUN finished. Best={self.last_result.best_cost:.2f}, gap={gap:+.2f}% "
                f"(best gen {self.last_result.best_generation})."
            )
        else:
            self.status_var.set(
                f"RUN finished. Best={self.last_result.best_cost:.2f} at gen={self.last_result.best_generation}."
            )

    def _run_multi(self) -> None:
        if self.current_instance is None:
            self.status_var.set("No instance loaded.")
            return
        try:
            base_config = self._build_ga_config_from_ui()
            experiments = int(self.num_experiments_var.get())
        except (ValueError, tk.TclError) as exc:
            self.status_var.set(f"Invalid MULTIRUN params: {exc}")
            return

        if experiments < 1:
            self.status_var.set("Number of experiments must be >= 1.")
            return

        best_result: GARunResult | None = None
        best_overall = float("inf")
        avg_best_costs: list[float] = []

        for run_idx in range(experiments):
            run_seed = base_config.seed + run_idx if base_config.seed is not None else None
            config = GAConfig(
                population_size=base_config.population_size,
                num_generations=base_config.num_generations,
                selection_method=base_config.selection_method,
                tournament_size=base_config.tournament_size,
                crossover_method=base_config.crossover_method,
                crossover_prob=base_config.crossover_prob,
                mutation_method=base_config.mutation_method,
                mutation_prob=base_config.mutation_prob,
                elitist=base_config.elitist,
                hillclimbing=base_config.hillclimbing,
                hillclimbing_variant=base_config.hillclimbing_variant,
                hillclimbing_start_generation=base_config.hillclimbing_start_generation,
                seed=run_seed,
            )

            self.status_var.set(f"MULTIRUN progress: {run_idx + 1}/{experiments}")
            self.root.update_idletasks()
            result = run_ga(instance=self.current_instance, config=config, progress_callback=None)
            avg_best_costs.append(result.best_cost)
            if result.best_cost < best_overall:
                best_overall = result.best_cost
                best_result = result

        if best_result is None:
            self.status_var.set("MULTIRUN failed to produce a result.")
            return

        self.last_result = best_result
        self.current_best_route = list(best_result.best_route)
        self.history_best = list(best_result.history_best)
        self.history_avg = list(best_result.history_avg)
        self.best_tour_label_var.set(f"{best_result.best_cost:.2f}")
        self.generation_label_var.set(str(base_config.num_generations))
        optimal_cost = self._read_optimal_cost_input()
        self.optimal_tour_label_var.set(f"{optimal_cost:.2f}" if optimal_cost is not None else "--")
        self._draw_instance_view()
        self._draw_convergence_chart()
        if optimal_cost is not None and optimal_cost > 0:
            gap = ((best_result.best_cost - optimal_cost) / optimal_cost) * 100.0
            self.status_var.set(
                f"MULTIRUN ({experiments}) done. Best={best_result.best_cost:.2f}, "
                f"mean best={mean(avg_best_costs):.2f}, gap={gap:+.2f}%."
            )
        else:
            self.status_var.set(
                f"MULTIRUN finished ({experiments} runs). Best={best_result.best_cost:.2f}, "
                f"mean best={mean(avg_best_costs):.2f}."
            )


def main() -> None:
    root = tk.Tk()
    try:
        TSPSimulatorUI(root)
    except Exception as exc:
        traceback.print_exc()
        tk.Label(
            root,
            text=f"GUI init error: {exc}",
            fg="#cc0000",
            bg="#f2f2f2",
            padx=12,
            pady=12,
            justify=tk.LEFT,
        ).pack(anchor=tk.NW)
    root.mainloop()


if __name__ == "__main__":
    main()
