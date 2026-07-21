# Smart Traffic Signal Controller Simulation

A Python desktop application with a visual UI that simulates a **Smart Traffic Signal Controller** managing a 4-way intersection. The app uses a priority-based AI optimizer for signal decisions, provides **Gemini LLM-powered explanations**, and includes a baseline comparison mode.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyQt5](https://img.shields.io/badge/UI-PyQt5-green)
![Gemini](https://img.shields.io/badge/AI-Gemini_API-blue)

---

## Features

- **Animated Intersection View** -- top-down 2D intersection view with vehicles as colored dots, live signal changes
- **Priority-Based Signal Optimizer** -- greedy need-score algorithm (`queue x wait x emergency_multiplier`)
- **Live KPI Charts** -- average wait time (line chart), vehicles cleared per direction (bar chart)
- **Gemini AI Explanations** -- natural-language explanations for every signal decision using the new `google-genai` SDK
- **Interactive Chat** -- ask follow-up questions about the controller's behavior
- **Baseline Comparison** -- side-by-side AI optimizer vs. fixed-timer results comparing 5 key performance indicators
- **Emergency Vehicle Mode** -- priority override for emergency vehicles
- **Multiple Scenarios** -- Light, Moderate, Heavy traffic presets, custom arrival rates, and custom CSV uploads
- **Setup Info** -- problem formulation dialog detailing system inputs, outputs, and constraints
- **Status Indicator Badge** -- prominent colored state indicator (IDLE, READY, SIMULATING, LOADING, ERROR)

---git remote set-url origin https://github.com/Aqsaazizwagan/TrafficControlSystem_6ED3CE.git

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set Gemini API Key

For AI-powered explanations, set your Gemini API key:

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "your-key-here"

# Linux / macOS
export GEMINI_API_KEY="your-key-here"
```

> **Note:** The app works fully without an API key -- it uses template-based explanations as a fallback.

### 3. Run the Application

```bash
python app.py
```

---

## How to Use

1. **Setup Info:** Click "Setup Info" to read the problem definition, inputs, outputs, and system constraints.
2. **Select a Scenario:** Select from the preset dropdown, adjust arrival rates with the sliders, or click "Load CSV" to upload a custom scenario file.
3. **Click Run** to start the simulation.
4. **Watch the intersection** animate with vehicles queuing and signals changing.
5. **Read AI explanations** in the right panel for each signal decision.
6. **Ask questions** in the chat box (e.g., "Why not extend East-West instead?").
7. **Pause / Resume / Reset** at any time.
8. **Compare to Baseline:** After the simulation completes, click "Compare to Baseline" to see side-by-side AI vs. fixed-timer graphs and tables.

---

## Project Structure

```
SmartTrafficController/
|-- app.py                  # Main UI application (PyQt5)
|-- traffic_engine.py       # Core simulation engine and PDF suggested wrappers
|-- ai_explainer.py         # Gemini API integration + fallback
|-- requirements.txt        # Python dependencies
|-- README.md               # This file
|-- AI_Lab_Project_Report.md # Full project documentation report
|-- data/
|   |-- presets.json        # Light/Moderate/Heavy scenario presets
|   |-- sample_scenario.csv # Pre-generated 300-tick scenario
|-- screenshots/            # UI screenshots folder
```

---

## AI Integration

### Search/Optimization AI (Core)
- Priority-based greedy scheduler computes `need_score = queue_length x avg_wait_time x emergency_multiplier`
- Selects the optimal green phase each tick
- Visualizes per-tick scores and decision factors

### NLP/LLM AI (Explanation Layer)
- **API:** Gemini API (`gemini-3.5-flash` with failover to `gemini-3.1-flash-lite`)
- **Input:** Current queue lengths, wait times, emergency flags, chosen phase (JSON)
- **Output:** 1-2 sentence plain-English explanation
- **Constraints:** Max 150 tokens/call, rate-limiting cooldown, environment variable for API key (never hardcoded)
- **Fallback:** Template-generated explanations when API is unavailable

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | PyQt5 |
| Charts | Matplotlib (embedded) |
| Core Algorithm | Pure Python (priority scheduler) |
| LLM Explanation | Gemini API (`google-genai`) |
| Data | NumPy, Pandas, CSV/JSON, PyMuPDF |

---
