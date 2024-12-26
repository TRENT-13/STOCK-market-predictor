import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support, silhouette_score
from sklearn.decomposition import PCA
import ta
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt


def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data.reset_index()
    return data

def calculate_technical_indicators(df):
    df = df.copy()

    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=8).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=13).ema_indicator()
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']
    ).volume_weighted_average_price()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    df['BB_upper'] = ta.volatility.BollingerBands(close=df['Close']).bollinger_hband()
    df['BB_lower'] = ta.volatility.BollingerBands(close=df['Close']).bollinger_lband()
    df['Stoch_K'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    #print(df.head(5))
    return df

def plot_elbow_silhouette(X_pca):
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    K = range(2, 10)
    distortions = []
    silhouette_scores = []

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_pca)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_pca, cluster_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(K, distortions, 'bx-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Distortion')
    ax1.set_title('The Elbow Method')

    if len(K) != len(silhouette_scores):
        silhouette_scores = silhouette_scores[:len(K)]
    ax2.plot(K, silhouette_scores, 'rx-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Scores')
    plt.tight_layout()
    plt.show()


    optimal_k = K[silhouette_scores.index(max(silhouette_scores))]

    return optimal_k

def create_labels(df, threshold=0.02):
    returns = df['Close'].pct_change()
    df['Label'] = 0
    df.loc[returns > threshold, 'Label'] = 1
    df.loc[returns < -threshold, 'Label'] = -1
    return df


def prepare_data(df, features):
    df = calculate_technical_indicators(df)
    df = create_labels(df)
    df = df.dropna()
    X = df[features].values
    y = df['Label'].values
    return X, y, df


def lorentzian_distance(x1, x2):
    return np.sum(np.log(1 + np.abs(x1 - x2)))




class LorentzianClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict_proba(self, X):
        probas = []
        for test_point in X:
            distances = [lorentzian_distance(test_point, train_point) for train_point in self.X_train]

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            nearest_distances = np.array(distances)[nearest_indices]

            weights = 1 / (nearest_distances + 1e-6)
            weights = weights / np.sum(weights)

            class_counts = np.zeros(3)
            for label, weight in zip(nearest_labels + 1, weights):
                class_counts[label] += weight

            probas.append(class_counts)
        return np.array(probas)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1) - 1




def plot_trading_performance(data, trades, title="Trading Performance"):
    plt.figure(figsize=(15, 7))
    plt.plot(data.index, data['Close'], label='Close Price', alpha=0.7)

    for trade in trades:
        entry_color = 'g' if trade['direction'] == 1 else 'r'
        exit_color = 'r' if trade['direction'] == 1 else 'g'
        marker = '^' if trade['direction'] == 1 else 'v'

        plt.scatter(data.index[trade['entry_date']], trade['entry_price'],
                   color=entry_color, marker=marker, s=100)
        plt.scatter(data.index[trade['exit_date']], trade['exit_price'],
                   color=exit_color, marker='x', s=100)

        plt.plot([data.index[trade['entry_date']], data.index[trade['exit_date']]],
                 [trade['entry_price'], trade['exit_price']],
                 color='gray', linestyle='--', alpha=0.5)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

#print res
def print_trading_summary(metrics):
    print("\nTrading Performance Summary:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Average Win: {metrics['avg_win']:.2%}")
    print(f"Average Loss: {metrics['avg_loss']:.2%}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Average Trade Duration: {metrics['avg_trade_duration']:.1f} days")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")


def plot_combined_analysis(data, cluster_labels, predictions, title="Stock Price with Clustering and Classification"):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Cluster'] = cluster_labels
    data['Prediction'] = predictions
    plt.figure(figsize=(15, 10))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='gray', alpha=0.6)
    clusters = data['Cluster'].unique() # error without distict
    cluster_colors = plt.cm.get_cmap("Set1", len(clusters))

    for cluster_id in clusters:
        cluster_data = data[data['Cluster'] == cluster_id]
        plt.scatter(cluster_data['Date'], cluster_data['Close'],
                    label=f'Cluster {cluster_id}',
                    color=cluster_colors(cluster_id),
                    alpha=0.6, s=50)

    buy_signals = data[data['Prediction'] == 1]
    sell_signals = data[data['Prediction'] == -1]

    plt.scatter(buy_signals['Date'], buy_signals['Close'],
                marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals['Date'], sell_signals['Close'],
                marker='v', color='red', s=100, label='Sell Signal')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_cluster_transitions(data, cluster_labels):
    data['Cluster'] = cluster_labels
    data['Returns'] = data['Close'].pct_change()

    # too advanced didnt get the reason till now
    data['Next_Cluster'] = data['Cluster'].shift(-1)

    transitions = []
    for i in range(len(data) - 1):
        if data['Cluster'].iloc[i] != data['Cluster'].iloc[i + 1]:
            transitions.append({
                'date': data['Date'].iloc[i],
                'from_cluster': data['Cluster'].iloc[i],
                'to_cluster': data['Cluster'].iloc[i + 1],
                'return': data['Returns'].iloc[i + 1]
            })

    #stats calc to print next, neededd
    if transitions:
        transition_df = pd.DataFrame(transitions)
        stats = {
            'total_transitions': len(transitions),
            'avg_return_after_transition': transition_df['return'].mean(),
            'positive_transitions': (transition_df['return'] > 0).sum() / len(transitions),
            'transition_matrix': pd.crosstab(
                transition_df['from_cluster'],
                transition_df['to_cluster']
            )
        }
    else:
        stats = {
            'total_transitions': 0,
            'avg_return_after_transition': 0,
            'positive_transitions': 0,
            'transition_matrix': pd.DataFrame()
        }

    return stats


def print_combined_analysis(trading_metrics, cluster_stats):

    print("\n=== Trading Strategy Performance ===")
    print(f"Total Trades: {trading_metrics['total_trades']}")
    print(f"Win Rate: {trading_metrics['win_rate']:.2%}")
    print(f"Total Return: {trading_metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.2f}")

    print("\n=== Clustering Analysis ===")
    print(f"Total Cluster Transitions: {cluster_stats['total_transitions']}")
    print(f"Average Return After Transition: {cluster_stats['avg_return_after_transition']:.2%}")
    print(f"Positive Transitions Rate: {cluster_stats['positive_transitions']:.2%}")

    print("\nTransition Matrix:")
    print(cluster_stats['transition_matrix'])


def calculate_trading_metrics(predictions, data):
    """
    Calculate trading metrics based solely on classification signals
    """
    trades = []
    position = None
    entry_price = None
    entry_date = None

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    for i in range(1, len(predictions)):
        current_price = data[i]
        signal = predictions[i]

        # Close position if signal changes
        if position is not None and signal != position:
            trades.append({
                'entry_date': entry_date,
                'exit_date': i,
                'direction': position,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': (current_price - entry_price) / entry_price * (1 if position == 1 else -1)
            })
            position = None
            entry_price = None
            entry_date = None

        # Open new position
        if position is None and signal != 0:
            position = signal
            entry_price = current_price
            entry_date = i

    # Close any remaining position
    if position is not None:
        pnl = (data[-1] - entry_price) / entry_price * (1 if position == 1 else -1)
        trades.append({
            'entry_date': entry_date,
            'exit_date': len(predictions) - 1,
            'direction': position,
            'entry_price': entry_price,
            'exit_price': data[-1],
            'pnl': pnl
        })

    if trades:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        total_pnl = sum(t['pnl'] for t in trades)
        trade_durations = [(t['exit_date'] - t['entry_date']) for t in trades]

        metrics = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(trades),
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if len(trades) > len(
                winning_trades) else 0,
            'total_return': total_pnl,
            'avg_trade_duration': np.mean(trade_durations),
            'max_drawdown': min(t['pnl'] for t in trades),
            'sharpe_ratio': np.mean([t['pnl'] for t in trades]) / np.std([t['pnl'] for t in trades]) if len(
                trades) > 1 else 0,
            'trades': trades
        }
    else:
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'total_return': 0,
            'avg_trade_duration': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'trades': []
        }

    return metrics

#for more 'aggresive model', trade do not stops till the opposite signal
# def calculate_trading_metrics(predictions, data):
#     """
#     Calculate trading metrics based on signal predictions with modified exit rules.
#     Only exits positions when an opposite signal appears (long exits on short signal,
#     short exits on long signal). Neutral signals are ignored for position exits.
#
#     Parameters:
#         predictions: Array of -1 (short), 0 (neutral), 1 (long) signals
#         data: Array of price data
#
#     Returns:
#         Dictionary containing trading metrics and list of trade details
#     """
#     trades = []
#     position = None  # Current position: None, 1 (long), or -1 (short)
#     entry_price = None
#     entry_date = None
#
#     if not isinstance(data, np.ndarray):
#         data = np.array(data)
#
#     for i in range(1, len(predictions)):
#         current_price = data[i]
#         signal = predictions[i]
#
#         # Case 1: No position is open
#         if position is None:
#             # Open a position if we get a directional signal
#             if signal != 0:  # If signal is long (1) or short (-1)
#                 position = signal
#                 entry_price = current_price
#                 entry_date = i
#
#         # Case 2: Long position is open
#         elif position == 1:
#             # Only exit if we get a short signal (-1)
#             # Ignore neutral signals (0)
#             if signal == -1:
#                 # Calculate profit/loss for long position
#                 pnl = (current_price - entry_price) / entry_price
#
#                 trades.append({
#                     'entry_date': entry_date,
#                     'exit_date': i,
#                     'direction': position,
#                     'entry_price': entry_price,
#                     'exit_price': current_price,
#                     'pnl': pnl,
#                     'duration': i - entry_date
#                 })
#
#                 # Immediately enter new short position
#                 position = signal
#                 entry_price = current_price
#                 entry_date = i
#
#         # Case 3: Short position is open
#         elif position == -1:
#             # Only exit if we get a long signal (1)
#             # Ignore neutral signals (0)
#             if signal == 1:
#                 # Calculate profit/loss for short position
#                 pnl = (entry_price - current_price) / entry_price
#
#                 trades.append({
#                     'entry_date': entry_date,
#                     'exit_date': i,
#                     'direction': position,
#                     'entry_price': entry_price,
#                     'exit_price': current_price,
#                     'pnl': pnl,
#                     'duration': i - entry_date
#                 })
#
#                 # Immediately enter new long position
#                 position = signal
#                 entry_price = current_price
#                 entry_date = i
#
#     # Handle any open position at the end of the period
#     if position is not None:
#         final_price = data[-1]
#         pnl = (final_price - entry_price) / entry_price if position == 1 else (entry_price - final_price) / entry_price
#
#         trades.append({
#             'entry_date': entry_date,
#             'exit_date': len(predictions) - 1,
#             'direction': position,
#             'entry_price': entry_price,
#             'exit_price': final_price,
#             'pnl': pnl,
#             'duration': len(predictions) - 1 - entry_date
#         })
#
#     # Calculate trading metrics
#     if trades:
#         # Calculate basic metrics
#         total_trades = len(trades)
#         winning_trades = [t for t in trades if t['pnl'] > 0]
#         total_pnl = sum(t['pnl'] for t in trades)
#         trade_durations = [t['duration'] for t in trades]
#
#         # Calculate returns for Sharpe ratio
#         trade_returns = [t['pnl'] for t in trades]
#
#         metrics = {
#             'total_trades': total_trades,
#             'winning_trades': len(winning_trades),
#             'win_rate': len(winning_trades) / total_trades,
#             'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
#             'avg_loss': np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if len(trades) > len(winning_trades) else 0,
#             'total_return': total_pnl,
#             'avg_trade_duration': np.mean(trade_durations),
#             'max_drawdown': min(trade_returns),
#             'sharpe_ratio': np.mean(trade_returns) / np.std(trade_returns) if len(trades) > 1 else 0,
#             'profit_factor': abs(sum(t['pnl'] for t in winning_trades) /
#                                sum(t['pnl'] for t in trades if t['pnl'] < 0)) if any(t['pnl'] < 0 for t in trades) else float('inf'),
#             'trades': trades
#         }
#     else:
#         # Return zero/empty metrics if no trades were made
#         metrics = {
#             'total_trades': 0,
#             'winning_trades': 0,
#             'win_rate': 0,
#             'avg_win': 0,
#             'avg_loss': 0,
#             'total_return': 0,
#             'avg_trade_duration': 0,
#             'max_drawdown': 0,
#             'sharpe_ratio': 0,
#             'profit_factor': 0,
#             'trades': []
#         }
#
#     return metrics


def plot_pca_clustering(X_pca, cluster_labels, title="PCA Clustering Results"):
    """
    Visualize clustering results in PCA space
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                         c=cluster_labels, cmap='viridis')

    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    ax.set_title(title)

    plt.colorbar(scatter)
    plt.show()



def plot_classification_results(data, title="Stock Price with Classification Results"):
        """
        Visualize stock price with classification signals (-1, 0, 1)
        """
        plt.figure(figsize=(15, 7))

        # Plot the stock price
        plt.plot(data['Date'], data['Close'], label='Close Price', color='gray', alpha=0.6)

        # Plot classification signals
        sell_signals = data[data['Prediction'] == -1]
        neutral_signals = data[data['Prediction'] == 0]
        buy_signals = data[data['Prediction'] == 1]

        # Plot each signal type with different markers and colors
        plt.scatter(buy_signals['Date'], buy_signals['Close'],
                    marker='^', color='green', s=100, label='Buy Signal (1)')
        plt.scatter(neutral_signals['Date'], neutral_signals['Close'],
                    marker='o', color='blue', s=50, label='Neutral Signal (0)')
        plt.scatter(sell_signals['Date'], sell_signals['Close'],
                    marker='v', color='red', s=100, label='Sell Signal (-1)')

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_results(y_true, y_pred, probabilities=None, pca=None, feature_names=None):
    """
    Create comprehensive visualization of model results including confusion matrix,
    ROC curves, and PCA feature importance for all three principal components.

    Parameters:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        probabilities: Array of prediction probabilities for each class
        pca: Fitted PCA object
        feature_names: List of feature names corresponding to the input data
    """
    plt.figure(figsize=(20, 15))

    plt.subplot(3, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    labels = ['-1', '0', '1']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if probabilities is not None:
        plt.subplot(3, 2, 2)
        colors = ['blue', 'green', 'red']
        class_labels = ['Negative', 'Neutral', 'Positive']

        for i in range(3):
            y_true_binary = (y_true == i - 1).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i],
                     label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend()

    if pca is not None and feature_names is not None:
        for i in range(3):
            plt.subplot(3, 2, i + 3)
            importance = np.abs(pca.components_[i])
            valid_importance = importance[:len(feature_names)]

            bars = plt.bar(feature_names, valid_importance)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Feature Importance (PCA Component {i + 1})')

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}',
                         ha='center', va='bottom')

        # Plot 6: Explained Variance
        plt.subplot(3, 2, 6)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(explained_variance) + 1), explained_variance,
                 marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.grid(True)

        for i, var in enumerate(explained_variance):
            plt.text(i + 1, var + 0.02, f'{var:.1%}',
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def visualize_stock_with_classification(stock_data, cluster_labels):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Cluster'] = cluster_labels

    plt.figure(figsize=(14, 7))

    plt.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='grey', linewidth=1.5)
    cluster_0_data = stock_data[stock_data['Cluster'] == 0]
    cluster_1_data = stock_data[stock_data['Cluster'] == 1]

    plt.scatter(cluster_1_data['Date'], cluster_1_data['Close'],
                label='Cluster 1', color='green', marker='o', alpha=0.8)
    plt.scatter(cluster_0_data['Date'], cluster_0_data['Close'],
                label='Cluster 0', color='red', marker='o', alpha=0.8)

    plt.title('Stock Price with Clustering Results')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_trade_distribution(trades):
    if not trades:  # so called error handling
        print("error, fix it")
        return

    returns = [t['pnl'] for t in trades]
    durations = [t['exit_date'] - t['entry_date'] for t in trades]

    plt.figure(figsize=(15, 5))

    # Returns distribution
    plt.subplot(1, 2, 1)
    plt.hist(returns, bins=20, edgecolor='black', color='skyblue')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Trade Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')

    # Duration distribution
    plt.subplot(1, 2, 2)
    plt.hist(durations, bins=20, edgecolor='black', color='skyblue')
    plt.title('Distribution of Trade Durations')
    plt.xlabel('Duration (days)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def main():
    symbol = 'GOOGL'
    data = fetch_stock_data(symbol, '2021-01-01', '2024-01-01')
    #unable to write df,column, cause do not have access, won't switch to modular because of that
    features = [
        'RSI', 'MACD', 'MACD_Signal', 'EMA_20', 'EMA_50', 'VWAP',
        'ATR', 'BB_upper', 'BB_lower', 'Stoch_K', 'OBV'
    ]

    X, y, processed_data = prepare_data(data, features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    optimal_k = plot_elbow_silhouette(X_pca)
    #kmeans = KMeans(n_clusters=optimal_k, random_state=42) depends siloute and elbow are just rule of thumb, sometimes hard to interpret
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)

    plot_pca_clustering(X_pca, cluster_labels)
    visualize_stock_with_classification(processed_data.copy(), cluster_labels)

    all_predictions = []
    all_probabilities = []
    tscv = TimeSeriesSplit(n_splits=5)

    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LorentzianClassifier(k=10)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)
        processed_data.loc[test_index, 'Prediction'] = predictions

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    plot_results(y[-len(all_predictions):], all_predictions,
                 probabilities=all_probabilities,
                 pca=pca,
                 feature_names=features)

    valid_data = processed_data.dropna(subset=['Prediction'])
    trading_metrics = calculate_trading_metrics(valid_data['Prediction'].values, valid_data['Close'].values)

    cluster_stats = analyze_cluster_transitions(processed_data, cluster_labels)

    plot_trading_performance(valid_data, trading_metrics['trades'])
    plot_trade_distribution(trading_metrics['trades'])
    print_trading_summary(trading_metrics)

    plot_classification_results(valid_data, "Classification Results Over Time")

    plot_combined_analysis(valid_data, cluster_labels[-len(valid_data):], valid_data['Prediction'].values)
    print_combined_analysis(trading_metrics, cluster_stats)

    valid_pred = valid_data['Prediction'].values

    return trading_metrics, valid_pred, trading_metrics['trades'], trading_metrics, cluster_stats


if __name__ == "__main__":
    trading_metrics, valid_pred, trades, metrics, cluster_stats = main()