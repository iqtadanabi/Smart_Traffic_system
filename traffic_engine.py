"""
traffic_engine.py — Smart Traffic Signal Controller Engine

Core simulation logic: Intersection model, Priority Optimizer,
Fixed-Timer Baseline, Simulation Runner, and Metrics Collector.

Fully decoupled from UI — communicates via callbacks/signals.
"""

import csv
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


# ────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────

@dataclass
class Vehicle:
    """A single vehicle in the simulation."""
    id: int
    lane: str               # "north", "south", "east", "west"
    arrival_tick: int
    is_emergency: bool = False
    wait_time: float = 0.0  # accumulated wait in ticks
    cleared: bool = False
    cleared_tick: int = -1


class SignalPhase(Enum):
    """Traffic signal phases for a 4-way intersection."""
    NORTH_SOUTH_GREEN = auto()
    EAST_WEST_GREEN = auto()
    YELLOW = auto()
    ALL_RED = auto()
    PEDESTRIAN = auto()


@dataclass
class PhaseDecision:
    """Record of a signal phase decision with scoring details."""
    tick: int
    chosen_phase: SignalPhase
    need_scores: dict          # {"north_south": float, "east_west": float}
    queue_lengths: dict        # {"north": int, "south": int, ...}
    avg_wait_times: dict       # {"north": float, "south": float, ...}
    emergency_flags: dict      # {"north": bool, "south": bool, ...}
    reason: str = ""           # short reason string


# ────────────────────────────────────────────────────────────
# Lane
# ────────────────────────────────────────────────────────────

class Lane:
    """Manages a queue of vehicles for one approach direction."""

    def __init__(self, name: str):
        self.name = name
        self.queue: deque[Vehicle] = deque()
        self.cleared_vehicles: list[Vehicle] = []
        self.total_arrivals = 0
        self.total_cleared = 0

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    @property
    def has_emergency(self) -> bool:
        return any(v.is_emergency for v in self.queue)

    @property
    def avg_wait_time(self) -> float:
        if not self.queue:
            return 0.0
        return sum(v.wait_time for v in self.queue) / len(self.queue)

    def add_vehicle(self, vehicle: Vehicle):
        self.queue.append(vehicle)
        self.total_arrivals += 1

    def clear_vehicles(self, count: int, current_tick: int) -> list[Vehicle]:
        """Clear up to `count` vehicles from the front of the queue."""
        cleared = []
        for _ in range(min(count, len(self.queue))):
            v = self.queue.popleft()
            v.cleared = True
            v.cleared_tick = current_tick
            self.cleared_vehicles.append(v)
            self.total_cleared += 1
            cleared.append(v)
        return cleared

    def tick_wait(self):
        """Increment wait time for all queued vehicles."""
        for v in self.queue:
            v.wait_time += 1.0

    def reset(self):
        self.queue.clear()
        self.cleared_vehicles.clear()
        self.total_arrivals = 0
        self.total_cleared = 0


# ────────────────────────────────────────────────────────────
# Intersection
# ────────────────────────────────────────────────────────────

class Intersection:
    """4-way intersection with traffic signal control."""

    DIRECTIONS = ["north", "south", "east", "west"]

    def __init__(self):
        self.lanes: dict[str, Lane] = {d: Lane(d) for d in self.DIRECTIONS}
        self.current_phase = SignalPhase.NORTH_SOUTH_GREEN
        self.phase_timer: int = 0           # ticks remaining in current phase
        self.previous_green: Optional[SignalPhase] = None
        self.next_green: Optional[SignalPhase] = None
        self.cycle_count: int = 0
        self.pedestrian_countdown: int = 0

    def get_green_lanes(self) -> list[str]:
        """Return which lanes currently have green."""
        if self.current_phase == SignalPhase.NORTH_SOUTH_GREEN:
            return ["north", "south"]
        elif self.current_phase == SignalPhase.EAST_WEST_GREEN:
            return ["east", "west"]
        return []

    def get_queue_snapshot(self) -> dict:
        return {d: self.lanes[d].queue_length for d in self.DIRECTIONS}

    def get_wait_snapshot(self) -> dict:
        return {d: round(self.lanes[d].avg_wait_time, 1) for d in self.DIRECTIONS}

    def get_emergency_snapshot(self) -> dict:
        return {d: self.lanes[d].has_emergency for d in self.DIRECTIONS}

    def reset(self):
        for lane in self.lanes.values():
            lane.reset()
        self.current_phase = SignalPhase.NORTH_SOUTH_GREEN
        self.phase_timer = 0
        self.previous_green = None
        self.next_green = None
        self.cycle_count = 0
        self.pedestrian_countdown = 0


# ────────────────────────────────────────────────────────────
# Constraints (loaded from presets.json or defaults)
# ────────────────────────────────────────────────────────────

@dataclass
class SimConstraints:
    min_green: int = 8
    yellow_duration: int = 3
    all_red_duration: int = 1
    pedestrian_window: int = 10
    pedestrian_cycle_interval: int = 4
    emergency_multiplier: float = 3.0
    phase_switch_threshold: float = 1.5
    vehicles_cleared_per_tick: int = 2  # how many vehicles pass per green tick

    @classmethod
    def from_presets_file(cls, path: str) -> "SimConstraints":
        try:
            with open(path, "r") as f:
                data = json.load(f)
            c = data.get("constraints", {})
            return cls(
                min_green=c.get("min_green_seconds", 8),
                yellow_duration=c.get("yellow_seconds", 3),
                all_red_duration=c.get("all_red_seconds", 1),
                pedestrian_window=c.get("pedestrian_window_seconds", 10),
                pedestrian_cycle_interval=c.get("pedestrian_cycle_interval", 4),
                emergency_multiplier=c.get("emergency_multiplier", 3.0),
                phase_switch_threshold=c.get("phase_switch_threshold", 1.5),
            )
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()


# ────────────────────────────────────────────────────────────
# Priority Optimizer
# ────────────────────────────────────────────────────────────

class PriorityOptimizer:
    """
    Greedy priority-based signal optimizer.
    
    Scores each approach group every tick using:
        need_score = queue_length × avg_wait_time × emergency_multiplier
    
    Selects the next green phase based on the highest-scoring group,
    subject to minimum green time constraints.
    """

    def __init__(self, constraints: SimConstraints):
        self.constraints = constraints

    def compute_need_score(self, intersection: Intersection, directions: list[str]) -> float:
        """Compute combined need score for a group of lanes."""
        total = 0.0
        for d in directions:
            lane = intersection.lanes[d]
            q = max(lane.queue_length, 0)
            w = max(lane.avg_wait_time, 0.1)  # avoid zero
            mult = self.constraints.emergency_multiplier if lane.has_emergency else 1.0
            total += q * w * mult
        return round(total, 2)

    def decide(self, intersection: Intersection, elapsed_in_phase: int) -> Optional[PhaseDecision]:
        """
        Decide whether to switch phase.
        
        Returns a PhaseDecision if a switch should happen, None otherwise.
        """
        ns_score = self.compute_need_score(intersection, ["north", "south"])
        ew_score = self.compute_need_score(intersection, ["east", "west"])

        scores = {"north_south": ns_score, "east_west": ew_score}
        queues = intersection.get_queue_snapshot()
        waits = intersection.get_wait_snapshot()
        emergencies = intersection.get_emergency_snapshot()

        current = intersection.current_phase

        # During yellow, all-red, or pedestrian — don't decide, just count down
        if current in (SignalPhase.YELLOW, SignalPhase.ALL_RED, SignalPhase.PEDESTRIAN):
            return None

        # Check if minimum green time has been served
        if elapsed_in_phase < self.constraints.min_green:
            return None

        # Determine opposing score
        if current == SignalPhase.NORTH_SOUTH_GREEN:
            current_score = ns_score
            opposing_score = ew_score
            candidate_phase = SignalPhase.EAST_WEST_GREEN
        else:
            current_score = ew_score
            opposing_score = ns_score
            candidate_phase = SignalPhase.NORTH_SOUTH_GREEN

        # Switch if opposing need is significantly higher
        should_switch = False
        reason = ""

        if opposing_score > current_score * self.constraints.phase_switch_threshold:
            should_switch = True
            reason = f"Opposing need ({opposing_score:.1f}) exceeds current ({current_score:.1f}) by {self.constraints.phase_switch_threshold}×"

        # Emergency override — switch immediately if opposing has emergency and current doesn't
        opposing_dirs = ["east", "west"] if current == SignalPhase.NORTH_SOUTH_GREEN else ["north", "south"]
        current_dirs = ["north", "south"] if current == SignalPhase.NORTH_SOUTH_GREEN else ["east", "west"]

        opposing_emergency = any(intersection.lanes[d].has_emergency for d in opposing_dirs)
        current_emergency = any(intersection.lanes[d].has_emergency for d in current_dirs)

        if opposing_emergency and not current_emergency:
            should_switch = True
            reason = "Emergency vehicle detected in opposing direction"

        if should_switch:
            return PhaseDecision(
                tick=0,  # will be set by runner
                chosen_phase=candidate_phase,
                need_scores=scores,
                queue_lengths=queues,
                avg_wait_times=waits,
                emergency_flags=emergencies,
                reason=reason,
            )

        return None


# ────────────────────────────────────────────────────────────
# Fixed Timer Controller (Baseline)
# ────────────────────────────────────────────────────────────

class FixedTimerController:
    """
    Simple fixed-timer baseline: 30s per green phase, round-robin.
    No intelligence — used for comparison with the AI optimizer.
    """

    FIXED_GREEN = 30

    def __init__(self, constraints: SimConstraints):
        self.constraints = constraints

    def decide(self, intersection: Intersection, elapsed_in_phase: int) -> Optional[PhaseDecision]:
        current = intersection.current_phase

        if current in (SignalPhase.YELLOW, SignalPhase.ALL_RED, SignalPhase.PEDESTRIAN):
            return None

        if elapsed_in_phase >= self.FIXED_GREEN:
            if current == SignalPhase.NORTH_SOUTH_GREEN:
                next_phase = SignalPhase.EAST_WEST_GREEN
            else:
                next_phase = SignalPhase.NORTH_SOUTH_GREEN

            return PhaseDecision(
                tick=0,
                chosen_phase=next_phase,
                need_scores={"north_south": 0, "east_west": 0},
                queue_lengths=intersection.get_queue_snapshot(),
                avg_wait_times=intersection.get_wait_snapshot(),
                emergency_flags=intersection.get_emergency_snapshot(),
                reason=f"Fixed timer: {self.FIXED_GREEN}s elapsed, rotating phase",
            )

        return None


# ────────────────────────────────────────────────────────────
# Metrics Collector
# ────────────────────────────────────────────────────────────

class MetricsCollector:
    """Tracks per-tick simulation metrics."""

    def __init__(self):
        self.avg_wait_history: list[float] = []
        self.throughput_per_direction: dict[str, int] = {
            "north": 0, "south": 0, "east": 0, "west": 0
        }
        self.tick_runtimes: list[float] = []
        self.total_cleared: int = 0
        self.total_wait: float = 0.0
        self.cleared_count_for_avg: int = 0
        
        # New KPIs for evaluation module alignment
        self.max_wait_time: float = 0.0
        self.total_emergency_wait: float = 0.0
        self.emergency_cleared_count: int = 0

    def record_tick(self, intersection: Intersection, runtime: float,
                    cleared_this_tick: dict[str, int]):
        """Record metrics for one simulation tick."""
        # Average wait of currently queued vehicles
        all_waits = []
        for lane in intersection.lanes.values():
            for v in lane.queue:
                all_waits.append(v.wait_time)
                # Track max wait of currently queued vehicles
                if v.wait_time > self.max_wait_time:
                    self.max_wait_time = v.wait_time
        avg = sum(all_waits) / len(all_waits) if all_waits else 0.0
        self.avg_wait_history.append(round(avg, 2))

        # Throughput
        for d, count in cleared_this_tick.items():
            self.throughput_per_direction[d] += count
            self.total_cleared += count

        self.tick_runtimes.append(runtime)

    def record_cleared_vehicle(self, vehicle: Vehicle):
        """Track cleared vehicle wait time for final average."""
        self.total_wait += vehicle.wait_time
        self.cleared_count_for_avg += 1
        
        # Track max wait of cleared vehicles
        if vehicle.wait_time > self.max_wait_time:
            self.max_wait_time = vehicle.wait_time
            
        # Track emergency vehicle wait times
        if vehicle.is_emergency:
            self.total_emergency_wait += vehicle.wait_time
            self.emergency_cleared_count += 1

    @property
    def overall_avg_wait(self) -> float:
        if self.cleared_count_for_avg == 0:
            return 0.0
        return round(self.total_wait / self.cleared_count_for_avg, 2)

    @property
    def overall_max_wait(self) -> float:
        return round(self.max_wait_time, 1)

    @property
    def overall_emergency_wait(self) -> float:
        if self.emergency_cleared_count == 0:
            return 0.0
        return round(self.total_emergency_wait / self.emergency_cleared_count, 2)

    @property
    def overall_throughput(self) -> int:
        return self.total_cleared

    @property
    def avg_tick_runtime(self) -> float:
        if not self.tick_runtimes:
            return 0.0
        return round(sum(self.tick_runtimes) / len(self.tick_runtimes) * 1000, 3)  # ms

    def reset(self):
        self.avg_wait_history.clear()
        self.throughput_per_direction = {d: 0 for d in ["north", "south", "east", "west"]}
        self.tick_runtimes.clear()
        self.total_cleared = 0
        self.total_wait = 0.0
        self.cleared_count_for_avg = 0
        self.max_wait_time = 0.0
        self.total_emergency_wait = 0.0
        self.emergency_cleared_count = 0

    def get_summary(self) -> dict:
        return {
            "avg_wait_time": self.overall_avg_wait,
            "max_wait_time": self.overall_max_wait,
            "emergency_wait_time": self.overall_emergency_wait,
            "total_throughput": self.overall_throughput,
            "throughput_per_direction": dict(self.throughput_per_direction),
            "avg_tick_runtime_ms": self.avg_tick_runtime,
            "avg_wait_history": list(self.avg_wait_history),
        }


# ────────────────────────────────────────────────────────────
# Scenario Loader & PDF Reference Functions
# ────────────────────────────────────────────────────────────

def load_preset_scenarios(path: str) -> dict:
    """Load preset scenarios from presets.json."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("scenarios", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_csv_scenario(path: str) -> list[dict]:
    """Load a pre-generated scenario from CSV."""
    rows = []
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "tick": int(row["tick"]),
                    "arrivals": {
                        "north": int(row["north_arrivals"]),
                        "south": int(row["south_arrivals"]),
                        "east": int(row["east_arrivals"]),
                        "west": int(row["west_arrivals"]),
                    },
                    "emergency": {
                        "north": bool(int(row["emergency_north"])),
                        "south": bool(int(row["emergency_south"])),
                        "east": bool(int(row["emergency_east"])),
                        "west": bool(int(row["emergency_west"])),
                    },
                })
    except (FileNotFoundError, csv.Error, KeyError, ValueError) as e:
        raise ValueError(f"Error reading CSV content: {e}")
    return rows


# --- PDF Reference Structure Functions ---

def load_data(path: str) -> list[dict]:
    """Wrapper function to load scenario dataset from a path (Required by Lab Project Guide)."""
    if path.endswith(".csv"):
        return load_csv_scenario(path)
    else:
        raise ValueError("Unsupported file format. Please upload a .csv file.")


def preprocess_data(data: list[dict]) -> list[dict]:
    """Validates the input scenario dataset format and values (Required by Lab Project Guide)."""
    if not isinstance(data, list):
        raise ValueError("Input data must be a list of tick entries.")
        
    if not data:
        raise ValueError("Scenario data is empty. Please provide at least one tick of data.")

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry at index {i} must be a dictionary.")
            
        # Validate tick
        if "tick" not in entry:
            raise ValueError(f"Entry at index {i} is missing required field 'tick'.")
        try:
            int(entry["tick"])
        except (ValueError, TypeError):
            raise ValueError(f"Tick value at index {i} must be an integer.")

        # Validate arrivals
        if "arrivals" not in entry or not isinstance(entry["arrivals"], dict):
            raise ValueError(f"Entry at index {i} must contain an 'arrivals' dictionary.")
        for d in ["north", "south", "east", "west"]:
            if d not in entry["arrivals"]:
                raise ValueError(f"Entry at index {i} 'arrivals' is missing lane '{d}'.")
            try:
                val = int(entry["arrivals"][d])
                if val < 0:
                    raise ValueError(f"Arrival count for '{d}' at tick {entry['tick']} cannot be negative.")
            except (ValueError, TypeError):
                raise ValueError(f"Arrival count for '{d}' at tick {entry['tick']} must be an integer.")

        # Validate emergency flags
        if "emergency" not in entry or not isinstance(entry["emergency"], dict):
            raise ValueError(f"Entry at index {i} must contain an 'emergency' dictionary.")
        for d in ["north", "south", "east", "west"]:
            if d not in entry["emergency"]:
                raise ValueError(f"Entry at index {i} 'emergency' is missing lane '{d}'.")
            if not isinstance(entry["emergency"][d], bool):
                try:
                    # Try to parse numeric boolean representation
                    entry["emergency"][d] = bool(int(entry["emergency"][d]))
                except (ValueError, TypeError):
                    raise ValueError(f"Emergency flag for '{d}' at tick {entry['tick']} must be a boolean.")

    return data


def run_model_or_algorithm(data: list[dict], params: SimConstraints, controller_type: str = "ai") -> dict:
    """Runs the core model simulation synchronously and returns metrics (Required by Lab Project Guide)."""
    # Validate parameters
    validated_data = preprocess_data(data)
    
    if controller_type == "ai":
        controller = PriorityOptimizer(params)
    else:
        controller = FixedTimerController(params)
        
    runner = SimulationRunner(validated_data, controller, params)
    summary = runner.run_all()
    return summary


def generate_scenario_data(
    arrival_rates: dict[str, float],
    duration: int,
    emergency_prob: float = 0.015,
    seed: int = 42
) -> list[dict]:
    """Generate synthetic scenario data using Poisson arrivals."""
    rng = random.Random(seed)
    data = []

    def poisson(lam):
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p < L:
                return k - 1

    for tick in range(1, duration + 1):
        arrivals = {}
        emergency = {}
        for d in ["north", "south", "east", "west"]:
            rate_per_tick = arrival_rates.get(d, 5) / 60.0
            arrivals[d] = poisson(rate_per_tick)
            emergency[d] = rng.random() < emergency_prob
        data.append({"tick": tick, "arrivals": arrivals, "emergency": emergency})

    return data


# ────────────────────────────────────────────────────────────
# Simulation Runner
# ────────────────────────────────────────────────────────────

class SimulationRunner:
    """
    Orchestrates tick-by-tick simulation execution.
    
    Decoupled from UI — uses callbacks for state updates:
      on_tick(tick, intersection, decision, metrics)
      on_phase_change(tick, old_phase, new_phase, decision)
      on_complete(metrics_summary)
    """

    def __init__(
        self,
        scenario_data: list[dict],
        controller=None,
        constraints: SimConstraints = None,
        emergency_mode: bool = True,
    ):
        self.scenario_data = scenario_data
        self.constraints = constraints or SimConstraints()
        self.controller = controller or PriorityOptimizer(self.constraints)
        self.emergency_mode = emergency_mode

        self.intersection = Intersection()
        self.metrics = MetricsCollector()

        self._vehicle_id_counter = 0
        self._current_tick = 0
        self._elapsed_in_phase = 0
        self._transition_timer = 0
        self._transition_target: Optional[SignalPhase] = None
        self._running = False
        self._paused = False
        self._completed = False

        # Callbacks
        self.on_tick: Optional[Callable] = None
        self.on_phase_change: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_decision: Optional[Callable] = None

        # Decision history
        self.decisions: list[PhaseDecision] = []

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def total_ticks(self) -> int:
        return len(self.scenario_data)

    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    @property
    def is_completed(self) -> bool:
        return self._completed

    def step(self) -> Optional[PhaseDecision]:
        """
        Execute a single simulation tick. Returns PhaseDecision if one was made.
        Call this from the UI timer loop.
        """
        if self._completed or self._current_tick >= len(self.scenario_data):
            self._completed = True
            if self.on_complete:
                self.on_complete(self.metrics.get_summary())
            return None

        tick_start = time.perf_counter()

        tick_data = self.scenario_data[self._current_tick]
        decision = None
        cleared_this_tick = {"north": 0, "south": 0, "east": 0, "west": 0}

        # 1) Spawn vehicles
        for d in Intersection.DIRECTIONS:
            count = tick_data["arrivals"].get(d, 0)
            is_emg = tick_data["emergency"].get(d, False) and self.emergency_mode
            for _ in range(count):
                self._vehicle_id_counter += 1
                v = Vehicle(
                    id=self._vehicle_id_counter,
                    lane=d,
                    arrival_tick=self._current_tick,
                    is_emergency=is_emg,
                )
                self.intersection.lanes[d].add_vehicle(v)

        # 2) Increment wait for all queued vehicles
        for lane in self.intersection.lanes.values():
            lane.tick_wait()

        # 3) Handle transition phases (yellow / all-red / pedestrian)
        if self._transition_timer > 0:
            self._transition_timer -= 1
            if self._transition_timer == 0:
                if self.intersection.current_phase == SignalPhase.YELLOW:
                    # Move to all-red
                    self.intersection.current_phase = SignalPhase.ALL_RED
                    self._transition_timer = self.constraints.all_red_duration
                elif self.intersection.current_phase == SignalPhase.ALL_RED:
                    # Check if pedestrian window needed
                    if (self.intersection.cycle_count > 0 and
                            self.intersection.cycle_count % self.constraints.pedestrian_cycle_interval == 0):
                        self.intersection.current_phase = SignalPhase.PEDESTRIAN
                        self._transition_timer = self.constraints.pedestrian_window
                    else:
                        # Switch to target green
                        old = self.intersection.current_phase
                        self.intersection.current_phase = self._transition_target
                        self._elapsed_in_phase = 0
                        self.intersection.cycle_count += 1
                        if self.on_phase_change:
                            self.on_phase_change(self._current_tick, old, self._transition_target, decision)
                elif self.intersection.current_phase == SignalPhase.PEDESTRIAN:
                    old = self.intersection.current_phase
                    self.intersection.current_phase = self._transition_target
                    self._elapsed_in_phase = 0
                    if self.on_phase_change:
                        self.on_phase_change(self._current_tick, old, self._transition_target, decision)
        else:
            # 4) Ask controller for a decision
            decision = self.controller.decide(self.intersection, self._elapsed_in_phase)
            if decision:
                decision.tick = self._current_tick
                self.decisions.append(decision)

                # Start yellow transition
                old_phase = self.intersection.current_phase
                self.intersection.previous_green = old_phase
                self._transition_target = decision.chosen_phase
                self.intersection.current_phase = SignalPhase.YELLOW
                self._transition_timer = self.constraints.yellow_duration

                if self.on_decision:
                    self.on_decision(decision)

            self._elapsed_in_phase += 1

        # 5) Clear vehicles on green lanes
        green_lanes = self.intersection.get_green_lanes()
        for d in green_lanes:
            cleared = self.intersection.lanes[d].clear_vehicles(
                self.constraints.vehicles_cleared_per_tick,
                self._current_tick
            )
            cleared_this_tick[d] = len(cleared)
            for v in cleared:
                self.metrics.record_cleared_vehicle(v)

        # 6) Record metrics
        tick_runtime = time.perf_counter() - tick_start
        self.metrics.record_tick(self.intersection, tick_runtime, cleared_this_tick)

        # 7) Callbacks
        if self.on_tick:
            self.on_tick(self._current_tick, self.intersection, decision, self.metrics)

        self._current_tick += 1

        # Check completion
        if self._current_tick >= len(self.scenario_data):
            self._completed = True
            if self.on_complete:
                self.on_complete(self.metrics.get_summary())

        return decision

    def run_all(self) -> dict:
        """Run entire simulation synchronously. Used for baseline comparison."""
        while not self._completed:
            self.step()
        return self.metrics.get_summary()

    def reset(self, scenario_data: list[dict] = None):
        """Reset the simulation for re-run."""
        if scenario_data:
            self.scenario_data = scenario_data
        self.intersection.reset()
        self.metrics.reset()
        self.decisions.clear()
        self._vehicle_id_counter = 0
        self._current_tick = 0
        self._elapsed_in_phase = 0
        self._transition_timer = 0
        self._transition_target = None
        self._running = False
        self._paused = False
        self._completed = False


# ────────────────────────────────────────────────────────────
# Quick self-test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate a small scenario and run it
    data = generate_scenario_data(
        arrival_rates={"north": 8, "south": 7, "east": 6, "west": 9},
        duration=100,
    )
    constraints = SimConstraints()
    runner = SimulationRunner(data, PriorityOptimizer(constraints), constraints)

    def on_decision(d):
        print(f"  Tick {d.tick}: -> {d.chosen_phase.name} | Scores: NS={d.need_scores['north_south']}, EW={d.need_scores['east_west']} | {d.reason}")

    runner.on_decision = on_decision
    summary = runner.run_all()

    print(f"\n=== Simulation Complete ===")
    print(f"Avg Wait Time: {summary['avg_wait_time']:.2f} ticks")
    print(f"Total Throughput: {summary['total_throughput']} vehicles")
    print(f"Avg Tick Runtime: {summary['avg_tick_runtime_ms']:.3f} ms")
    print(f"Throughput/direction: {summary['throughput_per_direction']}")
    print(f"Decisions made: {len(runner.decisions)}")
