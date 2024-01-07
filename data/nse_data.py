# data/nse_data.py
# This file contains the code to collect and preprocess the option chain data from the National Stock Exchange of India (NSE) website, using the web scraping and the pandas libraries, and store the data in a CSV file format.

# Import the libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the function to collect and preprocess the option chain data from the NSE website
def nse_data():
    # Define the URL of the NSE website
    url = "https://www.nseindia.com/option-chain"

    # Make a GET request to the URL and get the response
    response = requests.get(url)

    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        # Parse the response content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the table element that contains the option chain data
        table = soup.find("table", id="optionChainTable")

        # Check if the table element is not None
        if table is not None:
            # Read the table element using pandas and store it in a dataframe
            df = pd.read_html(str(table))[0]

            # Drop the unwanted columns and rows from the dataframe
            df = df.drop(columns=["Chart", "Open Interest", "Change in OI", "Volume", "Bid Qty", "Bid Price", "Ask Price", "Ask Qty"])
            df = df.dropna()

            # Rename the columns of the dataframe
            df.columns = ["call_ltp", "call_net_change", "call_iv", "strike_price", "put_iv", "put_net_change", "put_ltp"]

            # Add the underlying asset price, the interest rate, and the dividend rate columns to the dataframe
            # Assume the interest rate is 6% and the dividend rate is 0%
            df["underlying_price"] = soup.find("span", id="underlyValue").text
            df["interest_rate"] = 0.06
            df["dividend_rate"] = 0.00

            # Return the dataframe
            return df

        else:
            # Print an error message and return None
            print("Table element not found")
            return None

    else:
        # Print an error message and return None
        print("Response status code not OK")
        return None
