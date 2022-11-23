README

1. Files and Directory

This projects contains four code files:
    - Orlandi_Michele_module1.py: module with Reader object to get stock data
    - Orlandi_Michele_module2.py: module with Engineer object to create predictor variables
    - Orlandi_Michele_module3.py: module with Processor object to clean data and run models
    - Orlandi_Michele_project2.py: main file that runs the models and gets results

And three data files:
    - tickers.csv: file with 10 stock tickers
    - tickers_nasd.csv: file with NASDAQ stock information
    - tickers_nyse.csv: file with NYSE stock information

2. Packages and Classes

Python Version: 3.10.8

Libraries and Packages:
    - pandas
    - numpy
    - TA-lib
    - scikit-learn
    - plotly
    - yfinance
    - os
    - sys
    - gc
    - warnings
To install any of these libraries, in the terminal, run: pip install {library_name}

3. Compile and Run

In order to compile the file follow these steps:
    - store data files into a folder named "data"
    - store "data" folder and .py files into the same directory
    - in the terminal run: python3.10 Orlandi_Michele_project2.py

4. Output

This project outputs four results files:
    - small_NN_results.csv: NN results for small universe
    - large_NN_results.csv: NN results for large universe
    - small_SVC_results.csv: SVC results for small universe
    - large_SVC_results.csv: SVC results for large universe
The whole project will take about 2 hours to run