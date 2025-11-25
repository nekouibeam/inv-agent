from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

@tool
def search_news(query: str) -> str:
    """
    Searches for news and articles related to the query using DuckDuckGo.
    Useful for finding recent events, sentiment, and news about companies.
    """
    try:
        search = DuckDuckGoSearchResults(backend="news")
        results = search.invoke(query)
        return results
    except Exception as e:
        return f"Error searching news for {query}: {str(e)}"
