# data/cboe_data.py
# This file contains the code to collect and preprocess the option chain data from the Chicago Board Options Exchange (CBOE) website, using the requests and the pandas libraries, and store the data in a CSV file format.

# Import the libraries
import requests
import pandas as pd

# Define the function to collect and preprocess the option chain data from the CBOE website
def cboe_data():
    # Define the URL of the CBOE website
    url = "http://www.cboe.com/delayedquote/quote-table-download"

    # Define the parameters of the GET request
    params = {
        "ticker": "SPX", # The ticker symbol of the underlying asset
        "all": "on" # The option to download all available option contracts
    }

    # Make a GET request to the URL with the parameters and get the response
    response = requests.get(url, params=params)

    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        # Read the response content using pandas and store it in a dataframe
        df = pd.read_csv(response.content)

        # Drop the unwanted columns from the dataframe
        df = df.drop(columns=["Bid", "Ask", "Volume", "Open Int"])

        # Rename the columns of the dataframe
        df.columns = ["expiration_date", "strike_price", "call_put", "last"]

        # Pivot the dataframe to have separate columns for call and put option prices
        df = df.pivot_table(index=["expiration_date", "strike_price"], columns="call_put", values="last").reset_index()

        # Rename the columns of the dataframe
        df.columns = ["expiration_date", "strike_price", "call_ltp", "put_ltp"]

        # Add the underlying asset price, the interest rate, and the dividend rate columns to the dataframe
        # Assume the interest rate is 6% and the dividend rate is 0%
        df["underlying_price"] = df["strike_price"].mean() # Use the mean of the strike prices as an approximation of the underlying asset price
        df["interest_rate"] = 0.06
        df["dividend_rate"] = 0.00

        # Return the dataframe
        return df

    else:
        # Print an error message and return None
        print("Response status code not OK")
        return None
