# Option Chain Price Prediction

This program is a project to predict the option chain prices using different methods and strategies, such as the Black-Scholes model, the binomial tree model, the Monte Carlo simulation, the neural network method, the deep neural network method, the long short-term memory method, the attention mechanism method, the transformer method, and the proposed method. It also implements the hedging, the arbitrage, and the speculation strategies for option trading. It uses the pandas and the numpy libraries for data manipulation and calculation, and the matplotlib and the pandas libraries for data visualization and presentation.

## Installation

To run this program, you need to have Python 3 installed on your system. You also need to install the following libraries:

- pandas
- numpy
- matplotlib
- scipy
- tensorflow
- keras

You can install these libraries using the pip command:

bash
pip install pandas numpy matplotlib scipy tensorflow keras

## Usage

To run this program, you need to have a data file in CSV format, containing the following columns:

underlying_price: The price of the underlying asset
strike_price: The strike price of the option
interest_rate: The interest rate
dividend_rate: The dividend rate
expiration_date: The time to expiration of the option
call_ltp: The last traded price of the call option
put_ltp: The last traded price of the put option
call_iv: The implied volatility of the call option
put_iv: The implied volatility of the put option
You can use the sample data file provided in the data folder, or you can use your own data file.

To run this program, you need to execute the main.py file in the terminal, passing the name of the data file as an argument. For example:

python main.py data.csv

The program will run the different methods for option chain price prediction, compare their results, and display them in a graphical or tabular format. The program will also run the different strategies for option trading, and show their costs and profits. The program will save the results in the results folder, in CSV and PNG formats.
