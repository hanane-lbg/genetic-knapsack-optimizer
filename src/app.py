import random
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Project Selection Optimization", layout="wide", initial_sidebar_state="expanded")

st.title("💼 Project Selection Optimization Using Genetic Algorithm 🧬")

st.sidebar.header("⚙️ Configuration Panel")
num_projects = st.sidebar.number_input("Number of Projects", min_value=1, step=1, value=5)
budget = st.sidebar.number_input("Available Budget (Knapsack Capacity)", min_value=1, step=1, value=100)


profits_input = st.sidebar.text_area("Project Profits (Returns) (comma-separated)", "60, 100, 120, 80, 90")
weights_input = st.sidebar.text_area("Project Weights (Investment) (comma-separated)", "10, 20, 30, 15, 25")

uploaded_file = st.sidebar.file_uploader("Upload Project Data (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    profits = data["Profit"].tolist()
    weights = data["Weight"].tolist()
    num_projects = len(profits)
else:
    profits = list(map(int, profits_input.split(",")))
    weights = list(map(int, weights_input.split(",")))

st.sidebar.header("🔧 Genetic Algorithm Settings")
cycles = st.sidebar.number_input("Number of Generations", min_value=1, step=1, value=20)
population_size = st.sidebar.number_input("Population Size", min_value=5, step=5, value=50)
crossover_strategy = st.sidebar.selectbox("Crossover Strategy", ["Two-Point", "Single-Point", "Uniform"])
mutation_rate_method = st.sidebar.selectbox(
    "Mutation Rate Strategy",
    ["0.1 (Fixed)", "0.01 (Fixed)", "Lagrangian Method", "Relaxation Method"]
)


class Knapsack:
    def __init__(self, capacity, profits, weights):
        self.capacity = capacity
        self.profits = profits
        self.weights = weights
        self.num_projects = len(profits)

    def fitness(self, individual):
        total_weight = sum(ind * w for ind, w in zip(individual, self.weights))
        total_profit = sum(ind * p for ind, p in zip(individual, self.profits))
        return total_profit if total_weight <= self.capacity else 0


def select_parent_roulette(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return random.choices(population, weights=selection_probs, k=1)[0]

def crossover(parent1, parent2, strategy="Single-Point"):
    if strategy == "Single-Point":
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]
    elif strategy == "Two-Point":
        point1, point2 = sorted(random.sample(range(len(parent1)), 2))
        return parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    elif strategy == "Uniform":
        return [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]

def mutate(individual, mutation_rate=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]

def lagrangian_mutation_rate(gen, max_gens, fitnesses):
    return max(0.01, 1 / (1 + (sum(fitnesses) / max_gens) * (gen + 1)))

def relaxation_mutation_rate(gen, max_gens, fitnesses):
    fitness_range = max(fitnesses) - min(fitnesses)
    return max(0.01, 0.1 * (1 - gen / max_gens) * (1 / (1 + fitness_range)))

def genetic_algorithm(knapsack, population_size, cycles, mutation_rate_method, crossover_strategy):
    num_projects = knapsack.num_projects
    population = [[random.randint(0, 1) for _ in range(num_projects)] for _ in range(population_size)]
    all_fitness_data = []

    for gen in range(cycles):
        fitnesses = [knapsack.fitness(ind) for ind in population]
        new_population = []

        # Elitism: Preserve top 2 individuals
        elites = sorted(population, key=knapsack.fitness, reverse=True)[:2]
        new_population.extend(elites)

        for _ in range(population_size - len(elites)):
            parent1 = select_parent_roulette(population, fitnesses)
            parent2 = select_parent_roulette(population, fitnesses)
            child = crossover(parent1, parent2, strategy=crossover_strategy)

            # Determine mutation rate based on user selection
            if mutation_rate_method == "0.1 (Fixed)":
                mutation_rate = 0.1
            elif mutation_rate_method == "0.01 (Fixed)":
                mutation_rate = 0.01
            elif mutation_rate_method == "Lagrangian Method":
                mutation_rate = lagrangian_mutation_rate(gen, cycles, fitnesses)
            elif mutation_rate_method == "Relaxation Method":
                mutation_rate = relaxation_mutation_rate(gen, cycles, fitnesses)
            else:
                mutation_rate = 0.1  # Default to fixed 0.1

            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        all_fitness_data.append(fitnesses)

    best_solution = max(population, key=knapsack.fitness)
    best_fitness = knapsack.fitness(best_solution)
    return best_solution, best_fitness, all_fitness_data


if st.sidebar.button("🚀 Run Project Selection Optimization"):
    try:
        if len(profits) != len(weights) or len(profits) != num_projects:
            st.error("⚠️ Number of profits and weights must match the 'Number of Projects'!")
        else:
            knapsack = Knapsack(capacity=budget, profits=profits, weights=weights)
            best_solution, best_fitness, all_fitness_data = genetic_algorithm(
                knapsack, population_size, cycles, mutation_rate_method, crossover_strategy
            )

            # Results Section
            st.subheader("✨ Optimization Results")
            st.success(f"**Best Total Return on Investment (ROI):** {best_fitness}")
            st.write(f"**Best Solution (Selected Projects):** {best_solution}")

            # Selected Projects Details
            selected_projects = [
                {"Project": i + 1, "Profit (Return)": profits[i], "Investment Required": weights[i]}
                for i, val in enumerate(best_solution) if val == 1
            ]
            if selected_projects:
                df = pd.DataFrame(selected_projects)
                st.table(df)
            else:
                st.warning("No projects selected!")

            # Fitness Plot
            st.subheader("📊 Fitness Progression Over Generations")
            fitness_summary = [
                {"Generation": i + 1, "Max Fitness": max(gen), "Avg Fitness": sum(gen) / len(gen), "Min Fitness": min(gen)}
                for i, gen in enumerate(all_fitness_data)
            ]
            df_fitness_summary = pd.DataFrame(fitness_summary)

            # Fitness Plot
            fig, ax = plt.subplots(figsize=(4, 2 ))
            ax.plot(df_fitness_summary["Generation"], df_fitness_summary["Max Fitness"], label="Max Fitness", color="green")
            ax.plot(df_fitness_summary["Generation"], df_fitness_summary["Avg Fitness"], label="Avg Fitness", color="blue")
            ax.plot(df_fitness_summary["Generation"], df_fitness_summary["Min Fitness"], label="Min Fitness", color="red")
            ax.set_xlabel("Generation", fontsize=6)
            ax.set_ylabel("Fitness", fontsize=6)
            ax.set_title("Fitness Progression Across Generations", fontsize=6)
            ax.legend()
            st.pyplot(fig)

            # Fitness Summary Table
            st.subheader("📋 Fitness Summary")
            st.table(df_fitness_summary)

    except Exception as e:
        st.error(f"❌ Error: {e}")
st.sidebar.subheader("📌 Indications")
st.sidebar.markdown("""
    <style>
        .custom-markdown {
            font-size: 14px;
            color: #000000;  # Black color for text
        }
    </style>

    <div class="custom-markdown">
        <p><strong>How it works:</strong><br>  
        - <strong>Available Budget (Knapsack Capacity):</strong> The business's total investment capacity.<br>  
        - <strong>Projects:</strong> Each project has a required investment and an expected return.<br>  
        - <strong>Goal:</strong> Choose a combination of projects that maximizes the total return without exceeding the budget.</p>
    </div>
""", unsafe_allow_html=True)