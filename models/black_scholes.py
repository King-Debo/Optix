# models/black_scholes.py
# This file contains the code to implement the Black-Scholes model for option chain price prediction, using the NumPy and the SciPy libraries.

# Import the libraries
import numpy as np
import scipy.stats as stats

# Define the function to implement the Black-Scholes model
def black_scholes(data):
    # Get the input variables from the data
    S = data["underlying_price"] # The underlying asset price
    K = data["strike_price"] # The strike price
    r = data["interest_rate"] # The interest rate
    q = data["dividend_rate"] # The dividend rate
    T = data["expiration_date"] # The time to expiration
    sigma = data["call_iv"] # The implied volatility

    # Convert the input variables to numpy arrays
    S = np.array(S)
    K = np.array(K)
    r = np.array(r)
    q = np.array(q)
    T = np.array(T)
    sigma = np.array(sigma)

    # Calculate the intermediate variables
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N = stats.norm.cdf # The cumulative distribution function of the standard normal distribution

    # Calculate the call and put option prices
    C = S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)
    P = K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)

    # Return the call and put option prices
    return C, P
