# main.py
# This file contains the code to run the different methods for option chain price prediction, compare their results, and display them in a graphical or tabular format.

# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from data import load_data
from models import black_scholes, binomial_tree, monte_carlo, neural_network, deep_neural_network, long_short_term_memory, attention_mechanism, transformer
from strategies import hedging, arbitrage, speculation

# Load the data
data = load_data()

# Define the list of methods
methods = [black_scholes, binomial_tree, monte_carlo, neural_network, deep_neural_network, long_short_term_memory, attention_mechanism, transformer]

# Define the list of method names
method_names = ["Black-Scholes", "Binomial Tree", "Monte Carlo", "Neural Network", "Deep Neural Network", "Long Short-Term Memory", "Attention Mechanism", "Transformer", "Proposed Method"]

# Define the list of metrics
metrics = ["RMSE", "MAPE"]

# Define the list of option types
option_types = ["Call", "Put"]

# Initialize the dictionaries to store the results
results = {}
predictions = {}
hedge_costs = {}
arbitrage_profits = {}
speculation_profits = {}

# Loop through the methods
for i, method in enumerate(methods):
    # Run the method and get the predicted option prices
    C_pred, P_pred = method(data)
    # Store the predictions in the dictionary
    predictions[method_names[i]] = {"Call": C_pred, "Put": P_pred}
    # Calculate the actual option prices
    C_actual = data["call_ltp"]
    P_actual = data["put_ltp"]
    # Calculate the errors
    C_error = C_pred - C_actual
    P_error = P_pred - P_actual
    # Calculate the root mean squared error (RMSE)
    C_rmse = np.sqrt(np.mean(C_error ** 2))
    P_rmse = np.sqrt(np.mean(P_error ** 2))
    # Calculate the mean absolute percentage error (MAPE)
    C_mape = np.mean(np.abs(C_error / C_actual)) * 100
    P_mape = np.mean(np.abs(P_error / P_actual)) * 100
    # Store the results in the dictionary
    results[method_names[i]] = {"Call": {"RMSE": C_rmse, "MAPE": C_mape}, "Put": {"RMSE": P_rmse, "MAPE": P_mape}}
    # Run the hedging strategy and get the hedging cost
    C_cost, P_cost = hedging(data, predictions[method_names[i]])
    # Store the hedge costs in the dictionary
    hedge_costs[method_names[i]] = {"Call": C_cost, "Put": P_cost}
    # Run the arbitrage strategy and get the arbitrage profit
    profit = arbitrage(data, predictions[method_names[i]])
    # Store the arbitrage profits in the dictionary
    arbitrage_profits[method_names[i]] = profit
    # Run the speculation strategy and get the payoff and the profit
    C_payoff, C_profit, P_payoff, P_profit = speculation(data, predictions[method_names[i]])
    # Store the speculation profits in the dictionary
    speculation_profits[method_names[i]] = {"Call": {"Payoff": C_payoff, "Profit": C_profit}, "Put": {"Payoff": P_payoff, "Profit": P_profit}}

# Define the list of colors
colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive"]

# Plot the actual and predicted option prices for each method
plt.figure(figsize=(12, 8))
for i, method in enumerate(method_names):
    plt.scatter(data["call_ltp"], predictions[method]["Call"], color=colors[i], label=method)
plt.xlabel("Actual Call Option Price")
plt.ylabel("Predicted Call Option Price")
plt.title("Actual vs Predicted Call Option Price")
plt.legend()
plt.savefig("results/scatter_plots_call.png")
plt.show()

plt.figure(figsize=(12, 8))
for i, method in enumerate(method_names):
    plt.scatter(data["put_ltp"], predictions[method]["Put"], color=colors[i], label=method)
plt.xlabel("Actual Put Option Price")
plt.ylabel("Predicted Put Option Price")
plt.title("Actual vs Predicted Put Option Price")
plt.legend()
plt.savefig("results/scatter_plots_put.png")
plt.show()

# Create the data frames for the results
rmse_df = pd.DataFrame()
mape_df = pd.DataFrame()
p_value_df = pd.DataFrame()

# Loop through the metrics
for metric in metrics:
    # Loop through the option types
    for option_type in option_types:
        # Loop through the methods
        for method in method_names:
            # Get the result value
            value = results[method][option_type][metric]
            # Append the value to the data frame
            if metric == "RMSE":
                rmse_df.loc[option_type, method] = value
            elif metric == "MAPE":
                mape_df.loc[option_type, method] = value
        # Loop through the methods
        for i, method1 in enumerate(method_names):
            # Loop through the other methods
            for j, method2 in enumerate(method_names):
                # Skip if the methods are the same
                if i == j:
                    continue
                # Get the predicted option prices
                pred1 = predictions[method1][option_type]
                pred2 = predictions[method2][option_type]
                # Perform the paired t-test
                t_stat, p_value = ttest_rel(pred1, pred2)
                # Append the p-value to the data frame
                p_value_df.loc[option_type, f"{method1} vs {method2}"] = p_value

# Save the data frames as CSV files
rmse_df.to_csv("results/rmse.csv")
mape_df.to_csv("results/mape.csv")
p_value_df.to_csv("results/p_value.csv")

# Print the data frames
print("RMSE:")
print(rmse_df)
print("MAPE:")
print(mape_df)
print("p-value:")
print(p_value_df)

