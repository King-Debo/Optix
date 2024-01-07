# models/monte_carlo.py
# This file contains the code to implement the Monte Carlo simulation for option chain price prediction, using the NumPy library.

# Import the library
import numpy as np

# Define the function to implement the Monte Carlo simulation
def monte_carlo(data):
    # Get the input variables from the data
    S = data["underlying_price"] # The underlying asset price
    K = data["strike_price"] # The strike price
    r = data["interest_rate"] # The interest rate
    q = data["dividend_rate"] # The dividend rate
    T = data["expiration_date"] # The time to expiration
    sigma = data["call_iv"] # The implied volatility
    M = 1000 # The number of simulations

    # Convert the input variables to numpy arrays
    S = np.array(S)
    K = np.array(K)
    r = np.array(r)
    q = np.array(q)
    T = np.array(T)
    sigma = np.array(sigma)

    # Generate the standard normal random numbers
    Z = np.random.randn(M)

    # Simulate the underlying asset prices at expiration
    ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    # Calculate the call and put option payoffs at expiration
    CT = np.maximum(ST - K, 0)
    PT = np.maximum(K - ST, 0)

    # Calculate the call and put option prices by taking the average and discounting
    C = np.exp(-r * T) * np.mean(CT)
    P = np.exp(-r * T) * np.mean(PT)

    # Return the call and put option prices
    return C, P
