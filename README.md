# STOCK-market-predictor
# guide

## Core Data Functions

### fetch_stock_data(symbol, start_date, end_date)
This function serves as the data acquisition layer of our system. It uses the yfinance library to download historical stock data for a given symbol between specified dates. The function returns a pandas DataFrame containing essential price data like Open, High, Low, Close, and Volume, with the index reset to make the date a regular column.

### calculate_technical_indicators(df)
This function enriches the raw price data with various technical indicators that traders commonly use for analysis. Here's what each indicator tells us:

1. RSI (Relative Strength Index):
   - Measures momentum by comparing the magnitude of recent gains to recent losses
   - Range: 0-100, with values above 70 typically indicating overbought conditions and below 30 indicating oversold conditions

2. MACD (Moving Average Convergence Divergence):
   - Shows the relationship between two moving averages of the price
   - Generates: MACD line and Signal line
   - Helps identify trend changes and momentum

3. EMAs (Exponential Moving Averages):
   - EMA_20: Short-term trend (8-period)
   - EMA_50: Medium-term trend (13-period)
   - Gives more weight to recent prices than simple moving averages

4. VWAP (Volume Weighted Average Price):
   - Shows the average price weighted by volume
   - Important for identifying fair value and potential support/resistance levels

5. Additional Technical Indicators:
   - ATR (Average True Range): Measures volatility
   - Bollinger Bands: Shows potential support/resistance levels based on volatility
   - Stochastic Oscillator: Momentum indicator comparing closing price to price range
   - OBV (On Balance Volume): Cumulative volume indicator showing buying/selling pressure

## Analysis and Classification

### create_labels(df, threshold=0.02)
This function creates trading signals based on price movements:
- Returns > threshold: Label = 1 (Buy signal)
- Returns < -threshold: Label = -1 (Sell signal)
- Otherwise: Label = 0 (Hold/Neutral)

### prepare_data(df, features)
Orchestrates the data preparation process by:
1. Calculating technical indicators
2. Creating labels
3. Removing any rows with missing values
4. Extracting feature values and labels

### LorentzianClassifier
A custom classifier that uses the Lorentzian distance metric for prediction:

1. Key Methods:
   - fit(): Stores training data and labels
   - predict_proba(): Calculates class probabilities using k-nearest neighbors
   - predict(): Returns the most likely class for each input

2. Advantage of Lorentzian Distance:
   - Better handles outliers than Euclidean distance
   - More suitable for financial data which often has fat-tailed distributions

## Visualization Functions

### plot_results(y_true, y_pred, probabilities, pca, feature_names)
Creates a comprehensive visualization dashboard with four key plots:

1. Confusion Matrix:
   - Shows prediction accuracy across all classes
   - Helps identify where the model makes mistakes

2. ROC Curves:
   - Displays model performance at different classification thresholds
   - Shows trade-off between true positive and false positive rates

3. Feature Importance:
   - Based on PCA first component
   - Helps identify which indicators are most influential

4. Explained Variance:
   - Shows how much variance is captured by each principal component
   - Helps determine optimal dimensionality reduction

### visualize_stock_with_classification(stock_data, cluster_labels)
Creates a visual representation of the stock price with cluster assignments:
- Grey line: Base price movement
- Green points: Cluster 1 data points
- Red points: Cluster 0 data points
This helps identify market regimes and potential pattern changes.

### plot_trade_distribution(trades)
Visualizes the distribution of trading results:
1. Returns Distribution:
   - Shows the spread of profitable and unprofitable trades
   - Helps assess risk/reward characteristics

2. Duration Distribution:
   - Shows how long trades typically last
   - Helps optimize holding periods

## Performance Analysis

### calculate_trading_metrics(predictions, data)
Calculates comprehensive trading performance statistics:
- Win rate and total return
- Average win/loss size
- Trade durations
- Sharpe ratio and maximum drawdown
- Detailed trade log

### analyze_cluster_transitions(data, cluster_labels)
Studies how the market moves between different regimes:
- Tracks cluster changes over time
- Calculates return statistics for different transitions
- Creates a transition probability matrix

## Main Workflow

The main() function ties everything together in this sequence:

1. Data Preparation:
   - Fetches historical data
   - Calculates technical indicators
   - Prepares features and labels

2. Model Building:
   - Scales the data
   - Performs PCA for dimensionality reduction
   - Determines optimal number of clusters
   - Trains the Lorentzian classifier

3. Analysis:
   - Makes predictions using time series cross-validation
   - Calculates trading metrics
   - Analyzes cluster transitions

4. Visualization:
   - Creates various plots and charts
   - Prints performance summaries

This system combines multiple analytical approaches (technical analysis, clustering, and classification) to create a comprehensive trading analysis framework. The use of both clustering and classification helps identify market regimes while generating specific trading signals, potentially leading to more robust trading strategies.
