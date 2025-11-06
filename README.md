# 💼 Project Selection Optimization using Genetic Algorithms  

### 🚀 Advanced AI Mini Project — Based on the 0-1 Knapsack Problem  
  

---

## 📘 Overview  

This project implements a **Project Selection Optimization System** using **Genetic Algorithms (GAs)**, inspired by the **0-1 Knapsack Problem** in combinatorial optimization.  

The application allows decision-makers to select the best combination of projects to **maximize total return on investment (ROI)** while staying within a predefined **budget constraint**.  

The system is fully implemented as a **Streamlit web app**, enabling interactive visualization, parameter tuning, and optimization tracking in real-time.  

---

# 💼 Genetic Knapsack Optimizer

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bqehrmklhtiez3mca5natj.streamlit.app/)



## 🧠 Problem Definition  

Given a set of potential projects, each with:
- a **cost (weight)**  
- an **expected return (value)**  

The goal is to choose the optimal subset of projects that:  

\[
\text{Maximize } \sum_{i=1}^{n} v_i x_i \quad \text{subject to} \quad \sum_{i=1}^{n} w_i x_i \leq B, \quad x_i \in \{0, 1\}
\]

Where:
- \(v_i\): project return  
- \(w_i\): project cost  
- \(B\): total available budget  
- \(x_i\): decision variable (1 = select project *i*, 0 = skip it)  

This is structurally equivalent to the **0-1 Knapsack optimization problem**.  

---

## ⚙️ Algorithm Design  

### 🧬 Genetic Algorithm (GA)
A **metaheuristic approach** inspired by natural selection, consisting of:  

| Step | Description |
|------|--------------|
| **Initialization** | Generate an initial population of possible project selections |
| **Fitness Evaluation** | Calculate total ROI for each individual (solution) |
| **Selection** | Use **Roulette Wheel Selection** to favor high-ROI solutions |
| **Crossover** | Combine parent chromosomes using: Single-point, Two-point, or Uniform crossover |
| **Mutation** | Introduce diversity using: Bit-flip, Swap, or **Lagrangian Adaptive Mutation** |
| **Replacement** | Keep the fittest individuals for the next generation |
| **Stopping Criteria** | Stop when max generations or convergence is reached |

### 🧮 Adaptive Mutation (Proposed Enhancement)
Implemented a **Lagrangian-based dynamic mutation rate** to balance exploration and exploitation:  

\[
\text{Mutation Rate} = f(\text{generation number, fitness variance})
\]

This ensures robust convergence and avoids premature stagnation.  

---

## 💻 Application Features  

### 🧩 Streamlit Web App  

Interactive interface for configuring and solving the optimization problem.  

**Main Features:**  
- Upload or manually enter project dataset (name, cost, expected ROI)  
- Set budget and algorithm parameters (population size, mutation rate, generations, etc.)  
- Visualize algorithm convergence and best fitness value  
- Display selected projects and total optimized ROI  

**Tech Stack:**  
- 🐍 **Python**  
- 🧬 **NumPy**, **Pandas**, **Matplotlib**  
- 🌐 **Streamlit** for deployment and visualization  

---

## 📊 Example Result  

| Parameter | Value |
|------------|--------|
| Population Size | 100 |
| Generations | 100 |
| Budget | 50,000 |
| Mutation | Lagrangian Adaptive |
| Crossover | Single-Point |

**Best Solution:** ROI = **73,700**  

**Insights:**  
- Dynamic mutation achieved the best convergence performance.  
- Uniform crossover offered consistent balance between exploration and exploitation.  

---

