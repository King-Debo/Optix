# data/yahoo_data.py
# This file contains the code to collect and preprocess the option chain data from the Yahoo Finance API, using the requests and the pandas libraries, and store the data in a CSV file format.

# Import the libraries
import requests
import pandas as pd

# Define the function to collect and preprocess the option chain data from the Yahoo Finance API
def yahoo_data():
    # Define the URL of the Yahoo Finance API
    url = "https://query1.finance.yahoo.com/v7/finance/options/NFLX" # The URL for the option chain data of Netflix

    # Make a GET request to the URL and get the response
    response = requests.get(url)

    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        # Parse the response content as JSON and store it in a variable
        data = response.json()

        # Check if the data has the option chain key
        if "optionChain" in data:
            # Get the option chain value from the data
            option_chain = data["optionChain"]

            # Check if the option chain has the result key
            if "result" in option_chain:
                # Get the result value from the option chain
                result = option_chain["result"]

                # Check if the result is not empty
                if result:
                    # Get the first element of the result
                    result = result[0]

                    # Check if the result has the options key
                    if "options" in result:
                        # Get the options value from the result
                        options = result["options"]

                        # Check if the options is not empty
                        if options:
                            # Get the first element of the options
                            options = options[0]

                            # Check if the options has the calls and puts keys
                            if "calls" in options and "puts" in options:
                                # Get the calls and puts values from the options
                                calls = options["calls"]
                                puts = options["puts"]

                                # Convert the calls and puts values to dataframes
                                calls_df = pd.DataFrame(calls)
                                puts_df = pd.DataFrame(puts)

                                # Merge the calls and puts dataframes on the strike and expiration keys
                                df = pd.merge(calls_df, puts_df, on=["strike", "expiration"])

                                # Drop the unwanted columns from the dataframe
                                df = df.drop(columns=["contractSymbol_x", "contractSymbol_y", "lastTradeDate_x", "lastTradeDate_y", "change_x", "change_y", "percentChange_x", "percentChange_y", "volume_x", "volume_y", "openInterest_x", "openInterest_y", "inTheMoney_x", "inTheMoney_y", "contractSize_x", "contractSize_y", "currency_x", "currency_y", "impliedVolatility_x", "impliedVolatility_y"])

                                # Rename the columns of the dataframe
                                df.columns = ["strike_price", "call_ltp", "call_bid", "call_ask", "expiration_date", "put_ltp", "put_bid", "put_ask"]

                                # Add the underlying asset price, the interest rate, and the dividend rate columns to the dataframe
                                # Assume the interest rate is 6% and the dividend rate is 0%
                                df["underlying_price"] = result["quote"]["regularMarketPrice"] # Get the underlying asset price from the result
                                df["interest_rate"] = 0.06
                                df["dividend_rate"] = 0.00

                                # Return the dataframe
                                return df

                            else:
                                # Print an error message and return None
                                print("Calls and puts keys not found")
                                return None

                        else:
                            # Print an error message and return None
                            print("Options value is empty")
                            return None

                    else:
                        # Print an error message and return None
                        print("Options key not found")
                        return None

                else:
                    # Print an error message and return None
                    print("Result value is empty")
                    return None

            else:
                # Print an error message and return None
                print("Result key not found")
                return None

        else:
            # Print an error message and return None
            print("Option chain key not found")
            return None

    else:
        # Print an error message and return None
        print("Response status code not OK")
        return None