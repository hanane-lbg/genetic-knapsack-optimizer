
# 🧬 Project Selection using Genetic Algorithms (0-1 Knapsack)

> A Streamlit-powered web app that applies Genetic Algorithms to solve the 0-1 Knapsack problem. Users can optimize project selection under budget constraints to maximize return on investment.

## 🎯 Problem Statement

In many real-world scenarios like project funding or resource allocation, decision-makers must choose the best subset of items (projects) under a budget. This is modeled as a **0-1 Knapsack problem**, where each item has a profit and weight, and the objective is to **maximize total profit** without exceeding capacity.

This app allows you to:
- Upload your own dataset.
- Tune the **genetic algorithm parameters** (population size, crossover, mutation strategy).
- Visualize performance metrics across generations.

## 📸 Demo

![Streamlit UI](assets/streamlit_demo.png)

## 🧠 Algorithms Used

The solution leverages the following:
- **Genetic Algorithm (GA)**: Selection, crossover, mutation.
- **Mutation strategies**: Fixed rate, relaxation, Lagrangian method.
- **Crossover types**: Single-point, two-point, uniform.
- **Elitism**: Best individuals preserved in new generations.

## 🧪 Tech Stack

- Python 3.10+
- Streamlit
- Pandas
- Matplotlib

## ⚙️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/<username>/genetic-knapsack-optimizer.git
cd genetic-knapsack-optimizer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run src/app.py
```

## 📁 Example CSV format
```csv
Profit,Weight
60,10
100,20
120,30
80,15
90,25
```

## 📊 Sample Results

| Crossover     | Mutation     | Max Fitness |
|---------------|--------------|-------------|
| Single-Point  | 0.1 Fixed    | 73500       |
| Two-Point     | Lagrangian   | 73700       |
| Uniform       | Relaxation   | 73500       |

## 📄 Report

Find the full academic write-up in [`docs/0-1Knapsack.pdf`](docs/0-1Knapsack.pdf), which includes:
- Theoretical explanation of the KP01 problem.
- Description of algorithms.
- Comparative experiments.
- Interface screenshots.

## 👥 Authors

- Lebga Hanane, Abbou Riyad, Haggani Abla, Benounene Abdelrahmane  
> Supervised by: Ms. Mezrar, ENSI

## 📜 License

Licensed under the MIT License.
