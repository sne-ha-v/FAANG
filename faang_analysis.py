import streamlit as st
import logging
import aiohttp
import asyncio
import yfinance as yf
import pandas as pd
import json
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from crewai import Agent, Task, Process, Crew
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import contextlib

# Set page configuration
st.set_page_config(
    page_title="FAANG AI Stock Analysis",
    page_icon="📊",
    layout="wide"
)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define FAANG stocks and their full company names for news search
FAANG_COMPANIES = {
    "AAPL": "Apple Financial",
    "AMZN": "Amazon Financial",
    "META": "Meta Financial",
    "GOOGL": "Google Financial",
    "NFLX": "Netflix Financial"
}

# Function to fetch AI-related news and sentiment scores for each FAANG company
async def fetch_faang_news():
    url = "https://newsapi.org/v2/everything"
    api_key = "ac74a9a1b87a487c9603c0c0c0115c2b"
    
    results = {}

    async with aiohttp.ClientSession() as session:
        for ticker, query in FAANG_COMPANIES.items():
            params = {"q": query, "apiKey": api_key, "pageSize": 5, "language": "en"}
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        results[ticker] = {"Headlines": ["Error fetching news"], "Sentiment Score": 0}
                        continue
                    
                    data = await response.json()
                    if not data.get("articles"):
                        results[ticker] = {"Headlines": ["No recent news found"], "Sentiment Score": 0}
                        continue
                    
                    headlines = [article['title'] for article in data['articles']]
                    sentiment_scores = [sia.polarity_scores(headline)["compound"] for headline in headlines]
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

                    results[ticker] = {"Headlines": headlines, "Sentiment Score": round(avg_sentiment, 2)}
            
            except Exception as e:
                logging.error(f"Error fetching news for {ticker}: {e}")
                results[ticker] = {"Headlines": ["Error fetching news"], "Sentiment Score": 0}

    return results

# Function to fetch detailed stock data including Lag-1 analysis
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch 3 years of historical data
        hist = stock.history(period="3y")

        # Calculate Lag-1 differences
        latest_price = hist["Close"].iloc[-1]
        lag_1_day = hist["Close"].iloc[-2] if len(hist) > 1 else latest_price
        lag_1_month = hist["Close"].iloc[-22] if len(hist) > 22 else latest_price
        lag_1_year = hist["Close"].iloc[-252] if len(hist) > 252 else latest_price

        # Compute percentage changes
        day_change = round((latest_price - lag_1_day) / lag_1_day * 100, 2)
        month_change = round((latest_price - lag_1_month) / lag_1_month * 100, 2)
        year_change = round((latest_price - lag_1_year) / lag_1_year * 100, 2)

        # Additional financial data
        pe_ratio = stock.info.get("trailingPE", "N/A")
        market_cap = stock.info.get("marketCap", "N/A")
        eps = stock.info.get("trailingEps", "N/A")
        volume = hist["Volume"].iloc[-1] if "Volume" in hist.columns else "N/A"

        # Format market cap to billions
        if isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap / 1_000_000_000:.2f}B"

        result = {
            "Latest Price": round(latest_price, 2),
            "Lag-1 Day": f"{day_change}%",
            "Lag-1 Month": f"{month_change}%",
            "Lag-1 Year": f"{year_change}%",
            "P/E Ratio": pe_ratio,
            "Market Cap": market_cap,
            "EPS": eps,
            "Volume": volume
        }
        
        return result, hist
    
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return {}, pd.DataFrame()

# ARIMA Forecasting Function
def forecast_arima(stock_data, steps=5):
    model = ARIMA(stock_data['Close'], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast the next 'steps' values
    forecast = model_fit.forecast(steps=steps)
    
    return forecast, model_fit.resid

# Prepare LSTM model
def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Prepare LSTM Data using ARIMA residuals
def prepare_lstm_data(residuals, lookback=60):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(residuals.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Hybrid ARIMA-LSTM Forecasting
def hybrid_forecast(stock_data, steps=5):
    lookback = 60
    arima_forecast, residuals = forecast_arima(stock_data, steps)
    
    # Prepare LSTM data using residuals
    try:
        X_lstm, y_lstm, scaler = prepare_lstm_data(residuals, lookback)
        
        # Create and train LSTM model
        lstm_model = create_lstm_model((lookback, 1))
        lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=16, verbose=0)
        
        # Predict residuals for the next 'steps' time points
        last_residuals = residuals[-lookback:].values.reshape(-1, 1)
        last_residuals_scaled = scaler.transform(last_residuals)
        last_residuals_reshaped = last_residuals_scaled.reshape(1, lookback, 1)
        
        predicted_residuals = lstm_model.predict(last_residuals_reshaped)
        predicted_residuals = scaler.inverse_transform(predicted_residuals)
        
        # Final forecast = ARIMA forecast + predicted residuals
        final_forecast = arima_forecast.values + predicted_residuals.flatten()[0]
        
        return final_forecast, arima_forecast
    except Exception as e:
        st.warning(f"Error in hybrid forecasting. Using ARIMA only: {str(e)}")
        return arima_forecast.values, arima_forecast


# Define CrewAI agents
data_scientist = Agent(
    role="Data Scientist", 
    goal="Prepare and preprocess data for forecasting models like ARIMA and LSTM.",
    backstory="Expert in data preprocessing, feature engineering, and training machine learning models.",
    verbose=True
)

market_analyst = Agent(
    role="FINRA Approved Analyst", 
    goal="Analyze FAANG stock trends, financial news sentiment from VADERA, and historical lag analysis.",
    backstory="Expert in financial markets, specializing in stock trends and financial-driven market insights.",
    verbose=True
)

investment_consultant = Agent(
    role="Investment Advisor", 
    goal="Evaluate FAANG stock performance, sentiment scores, forecasting models, and recommend top investments.",
    backstory="Professional investment consultant with expertise in stock performance , financial market trends, and advanced forecasting techniques like ARIMA and LSTM.",
    verbose=True
)

stock_forecaster = Agent(
    role="Stock Forecaster", 
    goal="Use ARIMA and LSTM models to predict future stock prices for FAANG companies.",
    backstory="Specialist in stock price prediction using statistical models and deep learning techniques.",
    verbose=True
)

# Define tasks with logical dependencies
task1 = Task(
    description="Prepare and preprocess historical stock data for ARIMA and LSTM forecasting models.",
    agent=data_scientist,
    expected_output="Preprocessed data ready for forecasting with ARIMA and LSTM models, including cleaned historical price data, normalized features, and train-test splits."
)

task2 = Task(
    description="Fetch real-time stock data for FAANG stocks, including Lag-1 analysis and key financial metrics.",
    agent=market_analyst,
    expected_output="A comprehensive summary of FAANG stock prices, P/E ratios, Lag-1 changes, trading volumes, and recent market conditions."
)

task3 = Task(
    description="Analyze latest financial news sentiment for each FAANG company and compare it with stock trends over 3 years.",
    agent=market_analyst,
    expected_output="A detailed company-wise comparison of financial news sentiment, including sentiment scores, correlation with stock performance, and key trend analysis."
)

task4 = Task(
    description="Generate ARIMA and LSTM forecasting models for each FAANG stock using preprocessed data.",
    agent=stock_forecaster,
    expected_output="Detailed forecasting reports for each FAANG stock, including model performance metrics, predicted price ranges, and confidence intervals."
)

task5 = Task(
    description="""Generate a comprehensive investment recommendation based on:
    1. FAANG stock current data
    2. financial news sentiment analysis
    3. Lag-1 trends
    4. ARIMA and LSTM forecasting predictions
    
    Provide a detailed analysis comparing forecasting model predictions, sentiment scores, and current market conditions to identify the most promising FAANG stock for Q1 2025.""",
    agent=investment_consultant,
    expected_output="""A comprehensive investment recommendation for FAANG stocks in Q1 2025, including:
    1. Current market data and trends for each FAANG stock
    2. ARIMA and LSTM model forecasting results with a comparative analysis
    3. Financial news sentiment analysis and its impact on stock performance
    4. Final investment recommendation with a well-supported justification
    5. Risk assessment and potential market challenges"""
)



# Create CrewAI workflow
crew = Crew(
    agents=[data_scientist, market_analyst, stock_forecaster, investment_consultant],
    tasks=[task1, task2, task3, task4, task5],
    verbose=True,
    process=Process.sequential
)

# Main Streamlit App
def main():
    st.title("FAANG AI Stock Analysis Dashboard")
    st.markdown("### AI-Powered Stock Analysis with ARIMA-LSTM Hybrid Forecasting")
    
    # Create a progress bar for data loading
    progress_text = "Loading data and running AI analysis... Please wait."
    progress_bar = st.progress(0)
    
    # Run analysis in background
    stock_data_dict = {}
    historical_data_dict = {}
    forecasts_dict = {}
    arima_forecasts_dict = {}
    news_sentiment = {}
    
    # Run data collection
    with st.spinner("Fetching FAANG stock data and generating forecasts..."):
        for i, ticker in enumerate(FAANG_COMPANIES.keys()):
            # Update progress bar
            progress_bar.progress((i+1) / len(FAANG_COMPANIES))
            
            # Fetch stock data
            stock_info, historical_data = fetch_stock_data(ticker)
            stock_data_dict[ticker] = stock_info
            historical_data_dict[ticker] = historical_data
            
            # Generate forecasts only if we have historical data
            if not historical_data.empty:
                hybrid_forecast_result, arima_forecast_result = hybrid_forecast(historical_data, steps=5)
                forecasts_dict[ticker] = hybrid_forecast_result
                arima_forecasts_dict[ticker] = arima_forecast_result
        
        # Fetch news sentiment asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        news_sentiment = loop.run_until_complete(fetch_faang_news())
        loop.close()
    
    # Set progress to 100% when data loading is complete
    progress_bar.progress(100)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Current Data", "Historical Trends", "Forecasts", "Investment Advice"])
    
    # Tab 1: Current Data
    with tab1:
        st.header("Current FAANG Company Data")
        
        # Prepare DataFrame for display
        current_data = []
        for ticker in FAANG_COMPANIES.keys():
            if stock_data_dict[ticker]:  # Check if data exists
                row = {"Ticker": ticker}
                row.update(stock_data_dict[ticker])
                
                # Add sentiment data if available
                if ticker in news_sentiment:
                    sentiment_score = news_sentiment[ticker]["Sentiment Score"]
                    row["Sentiment"] = sentiment_score
                    
                    # Add color coding for sentiment
                    if sentiment_score > 0.2:
                        row["Sentiment Indicator"] = "🟢"  # Green for positive
                    elif sentiment_score < -0.2:
                        row["Sentiment Indicator"] = "🔴"  # Red for negative
                    else:
                        row["Sentiment Indicator"] = "🟡"  # Yellow for neutral
                
                current_data.append(row)
        
        # Create DataFrame
        if current_data:
            df_current = pd.DataFrame(current_data)
            
            # Show data table with styling
            st.dataframe(df_current.style.highlight_max(subset=["Latest Price", "Sentiment"], color="lightgreen")
                                        .highlight_min(subset=["Latest Price", "Sentiment"], color="lightcoral"), 
                        use_container_width=True)
            
            # Show recent headlines for the companies
            st.subheader("Recent Headlines")
            for ticker in FAANG_COMPANIES.keys():
                if ticker in news_sentiment and news_sentiment[ticker]["Headlines"]:
                    with st.expander(f"{ticker} - {FAANG_COMPANIES[ticker]} Headlines"):
                        for headline in news_sentiment[ticker]["Headlines"][:3]:
                            st.write(f"• {headline}")
        else:
            st.error("Failed to fetch current stock data.")
    
    # Tab 2: Historical Trends
    with tab2:
        st.header("FAANG Stock Historical Trends (3 Years)")
        
        # Create plot for historical data
        fig = go.Figure()
        
        # Add a stock selection dropdown
        selected_stocks = st.multiselect("Select stocks to display", 
                                         list(FAANG_COMPANIES.keys()), 
                                         default=list(FAANG_COMPANIES.keys()))
        
        for ticker in selected_stocks:
            if ticker in historical_data_dict and not historical_data_dict[ticker].empty:
                df = historical_data_dict[ticker]
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    mode="lines",
                    name=ticker
                ))
        
        fig.update_layout(
            title="FAANG Stock Price Trends (3-Year History)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend_title="Tickers",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a volume chart
        volume_fig = go.Figure()
        
        for ticker in selected_stocks:
            if ticker in historical_data_dict and not historical_data_dict[ticker].empty:
                df = historical_data_dict[ticker]
                if "Volume" in df.columns:
                    volume_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df["Volume"],
                        mode="lines",
                        name=f"{ticker} Volume"
                    ))
        
        volume_fig.update_layout(
            title="Trading Volume History",
            xaxis_title="Date",
            yaxis_title="Volume",
            legend_title="Tickers",
            hovermode="x unified"
        )
        
        st.plotly_chart(volume_fig, use_container_width=True)
    
    # Tab 3: Forecasts
    with tab3:
        st.header("Next 5 Days Price Forecasts")
        
        # Create date range for forecasts
        today = datetime.now()
        forecast_dates = [today + timedelta(days=i) for i in range(1, 6)]
        date_strings = [d.strftime("%Y-%m-%d") for d in forecast_dates]
        
        # Create combined forecast plot
        forecast_fig = go.Figure()
        
        for ticker in FAANG_COMPANIES.keys():
            if ticker in forecasts_dict and ticker in historical_data_dict:
                # Get historical data for context
                hist_data = historical_data_dict[ticker]
                if not hist_data.empty:
                    # Add historical data (last 30 days)
                    last_30_days = hist_data["Close"].iloc[-30:]
                    
                    # Add historical line
                    forecast_fig.add_trace(go.Scatter(
                        x=last_30_days.index,
                        y=last_30_days.values,
                        mode="lines",
                        name=f"{ticker} Historical",
                        line=dict(dash="solid")
                    ))
                    
                    # Add ARIMA forecast
                    if ticker in arima_forecasts_dict:
                        forecast_fig.add_trace(go.Scatter(
                            x=date_strings,
                            y=arima_forecasts_dict[ticker].values,
                            mode="lines+markers",
                            name=f"{ticker} ARIMA",
                            line=dict(dash="dot")
                        ))
                    
                    # Add Hybrid forecast
                    forecast_fig.add_trace(go.Scatter(
                        x=date_strings,
                        y=forecasts_dict[ticker],
                        mode="lines+markers",
                        name=f"{ticker} Hybrid",
                        line=dict(width=3)
                    ))
        
        forecast_fig.update_layout(
            title="FAANG Stock Price Forecasts (Next 5 Days)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend_title="Forecasts",
            hovermode="x unified"
        )
        
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Create a table to show forecast values
        st.subheader("Detailed Forecasts (Next 5 Days)")
        
        # Prepare forecast data for table display
        forecast_data = []
        
        for ticker in FAANG_COMPANIES.keys():
            if ticker in forecasts_dict:
                row = {"Ticker": ticker}
                
                # Add forecast values for each day
                for i, date in enumerate(date_strings):
                    row[f"Day {i+1} ({date})"] = round(forecasts_dict[ticker][i], 2)
                    
                # Calculate overall trend
                start_price = stock_data_dict[ticker].get("Latest Price", 0)
                end_price = forecasts_dict[ticker][-1]
                if start_price > 0:
                    change = ((end_price - start_price) / start_price) * 100
                    row["5-Day Forecast"] = f"{change:.2f}%"
                    
                    # Add direction indicator
                    if change > 3:
                        row["Trend"] = "🟢 Strong Up"
                    elif change > 0:
                        row["Trend"] = "🟢 Up"
                    elif change > -3:
                        row["Trend"] = "🟡 Flat"
                    else:
                        row["Trend"] = "🔴 Down"
                
                forecast_data.append(row)
        
        # Create forecast DataFrame
        if forecast_data:
            df_forecast = pd.DataFrame(forecast_data)
            st.dataframe(df_forecast, use_container_width=True)
    
    # Tab 4: Investment Advice
    with tab4:
            
        st.header("Investment Recommendation")
        with st.expander("View AI Agent Workflow Logs"):
                st.subheader("AI Agent Process Logs")
    
            # Create a placeholder for agent outputs
                agent_log = st.empty()
        
            # Run CrewAI with output redirection
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                result = crew.kickoff()
                agent_output = buf.getvalue()
        
            # Display the agent output in the Streamlit UI
                agent_log.code(agent_output)
        
if __name__ == "__main__":
    main()