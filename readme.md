# Time-Series Forecasting with SARIMA

This repository includes the implementation of SARIMA (Seasonal AutoRegressive Integrated Moving Average) model for time-series forecasting, illustrating both ARIMA and SARIMA model fitting.

## Project Structure

```
TIME-SERIES-FORECASTING-SARIMA
│
├── data
│   ├── train
│   │   └── train_data.csv
│   └── valid
│       └── valid_data.csv
│
├── myenv
│
├── notebook
│   └── ARIMA and SARIMA models.ipynb
│
├── scripts
│   └── time_series_sarima.py
│
├── .gitignore
│
└── requirements.txt
```

## Installation

Before running the scripts, make sure you have Python installed and the following packages:

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels

You can install all required packages using the following command:

```
pip install -r requirements.txt
```

## Usage

Navigate to the `notebook` directory and open the Jupyter notebook `ARIMA and SARIMA models.ipynb`. Execute the cells in sequence to preprocess the data, fit the ARIMA and SARIMA models, and visualize forecasts.

Alternatively, run the Python script `time_series_sarima.py` located in the `scripts` directory to execute the SARIMA model fitting and forecasting.

## Workflow

The main steps included in this time-series forecasting project are:

- Data loading and preprocessing.
- Visualization of train and validation data.
- Performing stationarity tests with Dickey-Fuller and KPSS.
- Differencing to achieve stationarity.
- Identification of ARIMA(p,d,q) model parameters using ACF and PACF plots.
- Fitting of ARIMA and SARIMA models.
- Forecasting future data points and comparing against a validation set.
- Visualization of the forecasting results.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your enhancements.

## License

This project is open source and available under standard MIT license terms.

```
