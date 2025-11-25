from langchain_core.tools import tool
import yfinance as yf

@tool
def get_stock_data(ticker: str) -> str:
    """
    Retrieves stock data for a given ticker symbol using yfinance.
    Returns a summary of price history (last 1 month) and basic info.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get history
        history = stock.history(period="1mo")
        if history.empty:
            return f"No price data found for {ticker}."
            
        # Get info (subset of important fields)
        info = stock.info
        info_summary = {
            "marketCap": info.get("marketCap"),
            "forwardPE": info.get("forwardPE"),
            "trailingPE": info.get("trailingPE"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "recommendationKey": info.get("recommendationKey")
        }
        
        # Format history summary (start, end, high, low, current)
        start_price = history.iloc[0]["Close"]
        end_price = history.iloc[-1]["Close"]
        high_price = history["High"].max()
        low_price = history["Low"].min()
        
        return f"""
        Ticker: {ticker}
        Info: {info_summary}
        Price Summary (Last 1 Month):
        - Start: {start_price:.2f}
        - End: {end_price:.2f}
        - High: {high_price:.2f}
        - Low: {low_price:.2f}
        
        Recent Daily Data (Last 5 days):
        {history.tail(5).to_string()}
        """
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"
