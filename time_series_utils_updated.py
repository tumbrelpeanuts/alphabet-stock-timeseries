import pandas as pd
from statsmodels.tsa.stattools import adfuller as adf, kpss
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Union
from tqdm.notebook import tqdm
import numpy as np


def adf_test(df):
    """
    Perform Augmented Dickey-Fuller (ADF) test on a time series DataFrame.

    Parameters:
    - df: Time series DataFrame.

    Returns:
    None (prints test results).
    """
    # Perform ADF test
    adf_result = adf(df)
    print("ADF Test Result:")
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value}")


# Example usage:
# Assuming you have a time series DataFrame named 'df'
# Call the function with your DataFrame
# adf_kpss_test(df)
# The code above is computing the ADF test and KPSS test
# * ADF test: checks for the presence of a unit root in the time series, which indicates non-stationarity. 
# * KPSS test: checks for stationarity around a deterministic trend, which is the opposite of the ADF test.
# I got the code from Selva from ML+. He gives the code if you sign up. Here is the link if you are interested: https://www.machinelearningplus.com/about-us/. Here is another one from [stack exchange](https://stats.stackexchange.com/questions/418997/augmented-dickey-fuller-test): https://stats.stackexchange.com/questions/418997/augmented-dickey-fuller-test


def plot_rolling_statistics(df, column: str, window: int = 12):
    """
    Compute and plot rolling mean and standard deviation.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column: Name of the column in the DataFrame for which rolling statistics will be computed.
    - window: Size of the rolling window.

    Returns:
    None (plots the rolling statistics).
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Compute rolling mean and standard deviation
    df_copy["rolling_avg"] = df_copy[column].rolling(window=window).mean()
    df_copy["rolling_std"] = df_copy[column].rolling(window=window).std()

    # Plot rolling statistics
    plt.figure(figsize=(15, 7))
    plt.plot(df_copy[column], color='#379BDB', label='Original')
    plt.plot(df_copy["rolling_avg"], color='#D22A0D', label='Rolling Mean')
    plt.plot(df_copy["rolling_std"], color='#142039', label='Rolling Std')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

# Example usage:
# Assuming adjusted_close_df is your DataFrame and 'Adjusted_Close' is the column name
# plot_rolling_statistics(adjusted_close_df, 'Adjusted_Close')


def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list) -> pd.DataFrame:
    """
    Optimize ARIMA parameters for a given endogenous time series data.

    Parameters:
    - endog: Endogenous time series data as a pandas Series or list.
    - order_list: List of tuples specifying the ARIMA orders to test.

    Returns:
    - result_df: DataFrame containing AIC values for each tested ARIMA order.
    """
    results = []
    
    for order in tqdm(order_list, desc='Fitting models'):
        try: 
            model = ARIMA(endog, order=(order[0], order[1], order[2])).fit()
        except:
            continue
            
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,d,q)', 'AIC']
    
    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df
# The code automates selecting the optimal ARIMA model by testing different order combinations and choosing the one with the lowest AIC. It creates a data frame of all possible results. The code was obtained from a book. The following link will link it to the creator's [GitHub](https://github.com/marcopeix/TimeSeriesForecastingInPython/blob/master/CH06/CH06.ipynb). Here is the link: https://github.com/marcopeix/TimeSeriesForecastingInPython/blob/master/CH06/CH06.ipynb


def optimize_SARIMA(endog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm(order_list):
        try: 
            model = SARIMAX(
                endog, 
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False,
            trend = 'c').fit(disp=False)
        except:
            continue
            
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df



def plot_sarima_train_predictions(test_df: pd.DataFrame, predictions) -> None:
    """
    This functions plots test set vs predicted values.
    ---
    Args:
        test_df (pd:DataFrame): 
        predictions (predictions): 
    
    Returns: None
    """
    # Figure
    fig, ax = plt.subplots(facecolor='w')

    ax.plot(np.exp(test_df), label='Training Set')
    ax.plot(np.exp(predictions), label='Forecast')

    # Labels
    ax.set_title("Train vs Predictions", fontsize=15, pad=10)
    ax.set_ylabel("Adjusted Stock Price", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    
    # Legend & Grid
    ax.grid(linestyle=":", color='grey')
    ax.legend()
    
    plt.show()


def plot_sarima_test_predictions(test_df: pd.DataFrame, predictions) -> None:
    """
    This functions plots test set vs predicted values.
    ---
    Args:
        test_df (pd:DataFrame): 
        predictions (predictions): 
    
    Returns: None
    """
    # Figure
    fig, ax = plt.subplots(facecolor='w')

    ax.plot(np.exp(test_df), label='Testing Set')
    ax.plot(np.exp(predictions), label='Forecast')

    # Labels
    ax.set_title("Test vs Predictions", fontsize=15, pad=10)
    ax.set_ylabel("Adjusted Stock Price", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    
    # Legend & Grid
    ax.grid(linestyle=":", color='grey')
    ax.legend()
    
    plt.show()


def plot_sarima_test_predictions_conf_in(test_df: pd.DataFrame, predictions, conf_int: pd.DataFrame) -> None:
    """
    This functions plots test set vs predicted values.
    ---
    Args:
        test_df (pd:DataFrame): 
        predictions (predictions): 
    
    Returns: None
    """
    # Figure
    fig, ax = plt.subplots(facecolor='w')

    ax.plot(np.exp(test_df), label='Testing Set')
    ax.plot(np.exp(predictions), label='Forecast')

    # Plotting the confidence intervals
    ax.fill_between(predictions.index, 
                    np.exp(conf_int.iloc[:, 0]), 
                    np.exp(conf_int.iloc[:, 1]), 
                    color='orange', alpha=0.1, label='Confidence Interval')

    # Labels
    ax.set_title("Test vs Predictions", fontsize=15, pad=10)
    ax.set_ylabel("Adjusted Stock Price", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    
    # Legend & Grid
    ax.grid(linestyle=":", color='grey')
    ax.legend()
    
    plt.show()


def plot_sarima_predictions(df_data: pd.DataFrame, predictions) -> None:
    """
    This function plots test set vs predicted values.
    ---
    Args:
        train_df (pd.DataFrame): The training set data.
        test_df (pd.DataFrame): The test set data.
        predictions (pd.Series): The forecasted values.
    
    Returns: None
    """
    # Figure
    fig, ax = plt.subplots(facecolor='w')

    ax.plot(df_data, label='Alphabet Stock Price')
    ax.plot(np.exp(predictions), label='Forecast', color='red')

    # Labels
    ax.set_title("Alphabet vs Prediction", fontsize=15, pad=10)
    ax.set_ylabel("Adjusted Stock Price", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    
    # Legend & Grid
    ax.grid(linestyle=":", color='grey')
    ax.legend()
    
    plt.show()



def optimize_SARIMAX(endog: Union[pd.Series, list], exog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
    results = []
    
    for order in tqdm(order_list):
        try:
            model = SARIMAX(
                endog,
                exog,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False).fit(disp=False)
        except:
            continue
            
        aic = model.aic
        # Calculate Heteroskedasticity P-Value directly from the fitted model
        #heteroskedasticity_pvalue = model.test_heteroskedasticity('breakvar')[1][1] if model.test_heteroskedasticity('breakvar') is not None else None
        #heteroskedasticity_pvalue = model.test_heteroskedasticity('breakvar')[0][1]
        results.append([order, aic])
        
    result_df = pd.DataFrame(results, columns=['(p,q,P,Q)', 'AIC'])
    
    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df



def plot_sarimax_train_predictions(target_train: pd.DataFrame, column: str, predictions: pd.Series) -> None:
    """
    This function plots the train set and  the forecast.
    ---
    Args:
        test_set (pd.DataFrame):  test set dataframe
        predictions (pd.Series):  forecast values as series
        
    Returns: None
    """
    # Figure
    fig, ax = plt.subplots(facecolor='w')

    ax.plot(np.exp(target_train[column]), label='Training Set')
    ax.plot(np.exp(predictions), label='Forecast')

    # Labels
    ax.set_title("Train vs Predictions", fontsize=15, pad=15)
    ax.set_ylabel("Adjusted Stock Price", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    
    # Legend & Grid
    ax.grid(linestyle=":", color='grey')
    ax.legend()
    
    plt.show()


def plot_sarimax_test_predictions(test_set: pd.DataFrame, column: str, predictions: pd.Series) -> None:
    """
    This function plots the train set and  the forecast.
    ---
    Args:
        test_set (pd.DataFrame):  test set dataframe
        predictions (pd.Series):  forecast values as series
        
    Returns: None
    """
    # Figure
    fig, ax = plt.subplots(facecolor='w')

    ax.plot(np.exp(test_set[column]), label='Testing Set')
    ax.plot(np.exp(predictions), label='Forecast')

    # Labels
    ax.set_title("Test vs Predictions", fontsize=15, pad=15)
    ax.set_ylabel("Adjusted Stock Price", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    
    # Legend & Grid
    ax.grid(linestyle=":", color='grey')
    ax.legend()
    
    plt.show()


def plot_sarimax_test_predictions_conf_in(test_set: pd.DataFrame, column: str, predictions: pd.Series, conf_int: pd.DataFrame) -> None:
    """
    This function plots the test set and the forecast with a confidence interval.
    ---
    Args:
        test_set (pd.DataFrame): Test set dataframe.
        column (str): Column name in the test set to plot.
        predictions (pd.Series): Forecast values as a series.
        conf_int (pd.DataFrame): Confidence intervals of the forecast.
        
    Returns: None
    """
    # Figure
    fig, ax = plt.subplots(facecolor='w')

    # Plotting the test set and forecast
    ax.plot(np.exp(test_set[column]), label='Testing Set')
    ax.plot(np.exp(predictions), label='Forecast')
    
    # Plotting the confidence intervals
    ax.fill_between(predictions.index, 
                    np.exp(conf_int.iloc[:, 0]), 
                    np.exp(conf_int.iloc[:, 1]), 
                    color='orange', alpha=0.1, label='Confidence Interval')

    # Labels
    ax.set_title("Test vs Predictions", fontsize=15, pad=15)
    ax.set_ylabel("Adjusted Stock Price", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    
    # Legend & Grid
    ax.grid(linestyle=":", color='grey')
    ax.legend()
    
    plt.show()



def plot_sarimax_predictions(df_data: pd.DataFrame, column: str, predictions: pd.Series) -> None:
    """
    This function plots the train set and  the forecast.
    ---
    Args:
        test_set (pd.DataFrame):  test set dataframe
        predictions (pd.Series):  forecast values as series
        
    Returns: None
    """
    # Figure
    fig, ax = plt.subplots(facecolor='w')
    
    ax.plot(df_data, label='Alphabet Stock Price')
    ax.plot(np.exp(predictions), label='Forecast', color='red')

    # Labels
    ax.set_title("Alphabet vs Predictions", fontsize=15, pad=15)
    ax.set_ylabel("Adjusted Stock Price", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    
    # Legend & Grid
    ax.grid(linestyle=":", color='grey')
    ax.legend()
    
    plt.show()


