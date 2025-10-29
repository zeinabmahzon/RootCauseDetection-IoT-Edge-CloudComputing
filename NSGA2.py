import sys
import subprocess
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

NUM_SENSORS = 50
MAX_INTERVALS = 20

ACCURACY_CACHE = {}

def get_prediction_accuracy(active_sensors: frozenset) -> float:
    if not active_sensors:
        return 0.0
    if active_sensors in ACCURACY_CACHE:
        return ACCURACY_CACHE[active_sensors]
    try:
        sensor_ids_str = [str(s_id) for s_id in active_sensors]
        command = [sys.executable, "XGB_evaluator.py"] + sensor_ids_str
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        accuracy = float(result.stdout.strip())
        ACCURACY_CACHE[active_sensors] = accuracy
        return accuracy
    except Exception as e:
        print(f"Error during subprocess call: {e}", file=sys.stderr)
        return 0.0

creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_interval", random.randint, 1, MAX_INTERVALS)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_interval, NUM_SENSORS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_schedule(individual):
    sensors_in_interval = {}
    for sensor_id, interval_id in enumerate(individual):
        sensors_in_interval.setdefault(interval_id, []).append(sensor_id)
    
    lifetime = len(sensors_in_interval)
    average_accuracy = 0.0
    
    if lifetime > 0:
        total_accuracy = sum(get_prediction_accuracy(frozenset(sensors)) for sensors in sensors_in_interval.values())
        average_accuracy = total_accuracy / lifetime
        print (lifetime, average_accuracy)
    return lifetime, average_accuracy

toolbox.register("evaluate", evaluate_schedule)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=MAX_INTERVALS, indpb=1.0/NUM_SENSORS)
toolbox.register("select", tools.selNSGA2)

def run_optimization_with_convergence_check():
    POP_SIZE = 100
    MAX_GENERATIONS = 500
    CX_PROB = 0.9
    MUT_PROB = 0.1
    
    PATIENCE = 25
    stall_counter = 0
    previous_front_fitnesses = set()

    pop = toolbox.population(n=POP_SIZE)
    hall_of_fame = tools.ParetoFront()
    
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hall_of_fame.update(pop)
    
    gen = 0
    while gen < MAX_GENERATIONS and stall_counter < PATIENCE:
        gen += 1
        
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        algorithms.varAnd(offspring, toolbox, CX_PROB, MUT_PROB)
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        hall_of_fame.update(offspring)
        
        pop = toolbox.select(pop + offspring, POP_SIZE)
        
        current_front_fitnesses = {ind.fitness.values for ind in hall_of_fame}
        
        if current_front_fitnesses == previous_front_fitnesses:
            stall_counter += 1
        else:
            stall_counter = 0
            previous_front_fitnesses = current_front_fitnesses

        print(f"Generation: {gen}, Stall Counter: {stall_counter}/{PATIENCE}, Pareto Front Size: {len(hall_of_fame)}")

    print(f"\nOptimization finished at generation {gen}.")
    if stall_counter >= PATIENCE:
        print(f"Reason: Pareto front has converged (no improvement for {PATIENCE} generations).")
    else:
        print("Reason: Reached maximum number of generations.")

    lifetimes = [sol.fitness.values[0] for sol in hall_of_fame]
    avg_accuracies = [sol.fitness.values[1] for sol in hall_of_fame]
    
    if not lifetimes:
        print("No solutions found in Pareto front. Cannot plot.")
        return

    sorted_points = sorted(zip(lifetimes, avg_accuracies))
    sorted_lifetimes, sorted_avg_accuracies = zip(*sorted_points)
    
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_lifetimes, sorted_avg_accuracies, 'bo-', label='Pareto-Optimal Solutions')
    plt.scatter(sorted_lifetimes, sorted_avg_accuracies, c='red')
    plt.title('Pareto-Optimal Front: Lifetime vs. Average Accuracy', fontsize=16)
    plt.xlabel('Network Lifetime (Number of Active Intervals)', fontsize=12)
    plt.ylabel('Average Prediction Accuracy', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_optimization_with_convergence_check()