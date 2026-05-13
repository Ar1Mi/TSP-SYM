import random
import traceback
import tkinter as tk
from tkinter import ttk


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
        self._draw_instance_placeholder()
        self._draw_convergence_placeholder()

    def _init_variables(self) -> None:
        self.instance_var = tk.StringVar(value="berlin52.tsp")
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

        self.use_seed_var = tk.BooleanVar(value=False)
        self.seed_value_var = tk.StringVar(value="123")

        self.num_experiments_var = tk.IntVar(value=1)

        self.generation_label_var = tk.StringVar(value="0")
        self.best_tour_label_var = tk.StringVar(value="--")
        self.optimal_tour_label_var = tk.StringVar(value="7542")
        self.status_var = tk.StringVar(value="Ready")

        self.show_best_in_gen_var = tk.BooleanVar(value=True)
        self.show_avg_in_gen_var = tk.BooleanVar(value=True)

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

    def _build_instance_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="TSP Instance", padding=8)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Read from file").grid(row=0, column=0, sticky="w")
        self.instance_combo = ttk.Combobox(
            frame,
            textvariable=self.instance_var,
            state="readonly",
            values=["berlin52.tsp", "eil51.tsp", "kroA100.tsp", "ch130.tsp"],
        )
        self.instance_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Checkbutton(
            frame,
            text="Show optimal path",
            variable=self.show_optimal_path_var,
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
            text="Clock seed",
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
            command=self._draw_convergence_placeholder,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            legend,
            text="Average in generation",
            variable=self.show_avg_in_gen_var,
            command=self._draw_convergence_placeholder,
        ).pack(side=tk.LEFT, padx=(12, 0))

    def _build_footer(self, parent: ttk.Frame) -> None:
        footer = ttk.Frame(parent)
        footer.pack(fill=tk.X, pady=(8, 0))
        footer.columnconfigure(0, weight=1)

        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        tk.Label(
            footer,
            text="Credits: Artur Gubanovich, Yahor Falkouski",
            fg="#888888",
            bg="#f2f2f2",
            font=("TkDefaultFont", 8),
        ).grid(row=0, column=1, sticky="e")

    def _wire_state_handlers(self) -> None:
        self.instance_canvas.bind("<Configure>", lambda _: self._draw_instance_placeholder())
        self.convergence_canvas.bind("<Configure>", lambda _: self._draw_convergence_placeholder())
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

    def _draw_instance_placeholder(self) -> None:
        canvas = self.instance_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 100)
        height = max(canvas.winfo_height(), 100)

        margin = 20
        usable_w = max(width - 2 * margin, 20)
        usable_h = max(height - 2 * margin, 20)

        random.seed(42)
        points = [
            (
                margin + random.random() * usable_w,
                margin + random.random() * usable_h,
            )
            for _ in range(45)
        ]

        for px, py in points:
            canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill="#52b7b0", outline="")

        if self.show_optimal_path_var.get() and len(points) > 2:
            ordered = sorted(points, key=lambda p: p[0] + p[1] * 0.2)
            for i in range(len(ordered) - 1):
                x1, y1 = ordered[i]
                x2, y2 = ordered[i + 1]
                canvas.create_line(x1, y1, x2, y2, fill="#83c9c4")

    def _draw_convergence_placeholder(self) -> None:
        canvas = self.convergence_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 220)
        height = max(canvas.winfo_height(), 180)
        left = 40
        bottom = height - 30
        right = width - 20
        top = 20

        canvas.create_line(left, bottom, right, bottom, fill="#666666")
        canvas.create_line(left, bottom, left, top, fill="#666666")
        canvas.create_text((left + right) / 2, bottom + 18, text="Generation")
        canvas.create_text(18, (top + bottom) / 2, text="Tour\nlength")

        series_len = 20
        xs = [left + (right - left) * i / (series_len - 1) for i in range(series_len)]

        if self.show_best_in_gen_var.get():
            best = [0.95 * (0.87**i) + 0.08 for i in range(series_len)]
            best_points = []
            for x, y in zip(xs, best):
                py = top + (bottom - top) * y
                best_points.extend((x, py))
            canvas.create_line(*best_points, fill="#4d79ff", width=2, smooth=True)

        if self.show_avg_in_gen_var.get():
            avg = [1.05 * (0.91**i) + 0.12 for i in range(series_len)]
            avg_points = []
            for x, y in zip(xs, avg):
                py = top + (bottom - top) * y
                avg_points.extend((x, py))
            canvas.create_line(*avg_points, fill="#cc4f79", width=2, smooth=True)

    def _run_once(self) -> None:
        self.generation_label_var.set("1")
        self.best_tour_label_var.set(str(random.randint(7300, 9800)))
        self.status_var.set("RUN pressed (placeholder). Algorithm wiring is next.")
        self._draw_instance_placeholder()
        self._draw_convergence_placeholder()

    def _run_multi(self) -> None:
        self.generation_label_var.set(str(self.num_generations_var.get()))
        self.best_tour_label_var.set(str(random.randint(7000, 9100)))
        self.status_var.set(
            f"MULTIRUN pressed for {self.num_experiments_var.get()} experiment(s) (placeholder)."
        )
        self._draw_instance_placeholder()
        self._draw_convergence_placeholder()


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
