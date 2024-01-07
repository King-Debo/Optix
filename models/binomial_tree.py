# models/binomial_tree.py
# This file contains the code to implement the binomial tree model for option chain price prediction, using the NumPy library.

# Import the library
import numpy as np

# Define the function to implement the binomial tree model
def binomial_tree(data):
    # Get the input variables from the data
    S = data["underlying_price"] # The underlying asset price
    K = data["strike_price"] # The strike price
    r = data["interest_rate"] # The interest rate
    q = data["dividend_rate"] # The dividend rate
    T = data["expiration_date"] # The time to expiration
    sigma = data["call_iv"] # The implied volatility
    N = 100 # The number of steps in the binomial tree

    # Convert the input variables to numpy arrays
    S = np.array(S)
    K = np.array(K)
    r = np.array(r)
    q = np.array(q)
    T = np.array(T)
    sigma = np.array(sigma)

    # Calculate the intermediate variables
    dt = T / N # The time interval
    u = np.exp(sigma * np.sqrt(dt)) # The up factor
    d = 1 / u # The down factor
    p = (np.exp((r - q) * dt) - d) / (u - d) # The risk-neutral probability
    df = np.exp(-r * dt) # The discount factor

    # Initialize the arrays to store the option prices at each node
    C = np.zeros((N + 1, N + 1)) # The call option prices
    P = np.zeros((N + 1, N + 1)) # The put option prices

    # Calculate the option prices at the final nodes
    for i in range(N + 1):
        C[i, N] = max(S * u ** (N - i) * d ** i - K, 0)
        P[i, N] = max(K - S * u ** (N - i) * d ** i, 0)

    # Backward induction to calculate the option prices at the previous nodes
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            C[i, j] = df * (p * C[i, j + 1] + (1 - p) * C[i + 1, j + 1])
            P[i, j] = df * (p * P[i, j + 1] + (1 - p) * P[i + 1, j + 1])

    # Return the call and put option prices at the root node
    return C[0, 0], P[0, 0]
