# **Table of Contents**

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [License](#license)
- [Citation](#citation)

---

# **Overview**

This repository implements ε-greedy reinforcement learning algorithms for policy optimization in complex system dynamics models. The research focuses on applying RL techniques to enhance decision-making in:

- Lotka-Volterra predator-prey dynamics  
- World2 global sustainability model

---

# **Project Structure**

```
RL_SD/
├── src/                          # Main algorithm implementations
│   ├── egreedy_lotkavolterra.py  # RL agent for Lotka–Volterra model
│   └── egreedy_world2.py         # RL agent for World2 model
│
├── report_helpers/               # Visualization and analysis tools
│   ├── plotting.py               # Learning curve visualizations
│   └── timing.py                 # Performance timing analysis
│
├── models/                       # System dynamics models
│   ├── pyworld2/                 # World2 model
│   ├── LotkaVolterra.py          # Lotka–Volterra model
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation (this file)
```
---

# **How to Run**

Follow these steps to set up the environment and execute the experiments.

### **1. Clone the Repository**
```bash
git clone https://github.com/juancamiloespana/RL_DS
cd RL_SD
```

### **2. Create and Activate a Virtual Environment**
It is recommended to use a dedicated virtual environment to avoid dependency conflicts.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### **4. Run the Reinforcement Learning Experiments**
Execute the training modules directly from the project root:

**Lotka–Volterra model:**
```bash
python -m src.egreedy_lotkavolterra
```

**World2 sustainability model:**
```bash
python -m src.egreedy_world2
```

### **5. View Results**
All experiment outputs are automatically saved in the root directory of the project (`RL_DS`). See [Output Files](#output-files) for details.

**Note:** You can modify experiment parameters directly in the scripts inside the `src/` folder to explore different learning behaviors.

---

## **Output Files**

### **1. Experimental Results CSV**
**Files:**
- `RL_LotkaVolterra_Experiment_Results.csv` (Lotka-Volterra experiments)
- `RL_World2_Experiment_Results.csv` (World2 experiments)

**Structure:**
| Column | Description |
|--------|-------------|
| `Epsilon_Level` | Exploration rate parameter (ε) |
| `Rho_Level` | Parameter adjustment factor (ρ) - percentage modification |
| `Repetition` | Experiment repetition number |
| `Run` | Iteration number |
| `Execution_Time` | Computational time (seconds) |
| `Return` | Cumulative reward obtained |

**Purpose:** Contains raw experimental data for all treatment combinations, enabling reproducibility and further statistical analysis.


### **2. Execution Time Statistics**
**File:** `[Experiment_Name]_Timing_Statistics.csv`

**Structure:**
| Column | Description |
|--------|-------------|
| `Epsilon_Level` | Exploration rate parameter (ε) |
| `Rho_Level` | Parameter adjustment factor (ρ) - percentage modification |
| `Mean` | Average execution time across repetitions |
| `Std` | Standard deviation of execution time |

**Purpose:** Performance benchmarking across different hyperparameter configurations.


##### **3. Learning Curve Visualizations**
**Format:** Interactive HTML plots (Plotly)

**Features:**
- **Mean return trajectories** – Average performance across repetitions
- **95% Confidence intervals** – Statistical uncertainty bands (2.5th - 97.5th percentiles)
- **Treatment comparison** – Multiple (ε, ρ) configurations overlaid

---

## **Licence**
This code is freely available for academic and research purposes. If you use this code in your research, please cite this repository.

---

## **Citation**

If you use this code in your research, please cite this repository:

```bibtex
@misc{rl_ds_2025,
  author = {Juan C. España and Esteffany Peña-Puentes and Carlos Enrique Vásquez-Ortiz and Sebastián Jaén},
  title = {RL\_DS: Reinforcement Learning and System Dynamics for policy optimization},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/juancamiloespana/RL_DS}},
  note = {GitHub repository}
}
```
