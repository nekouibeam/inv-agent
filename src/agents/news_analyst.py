from langchain.agents import create_agent
from ..state import AgentState
from ..tools.search_tools import search_news
from ..utils import get_llm

def news_analyst_node(state: AgentState):
    """
    Finance News Analyst that searches for and summarizes news using a ReAct agent.
    """
    llm = get_llm(temperature=0)
    tools = [search_news]
    
    system_prompt = """You are a Finance News Analyst.
    Your goal is to find and summarize relevant news for the provided tickers.
    
    1. Use the `search_news` tool to find recent news for EACH ticker.
    2. Focus on market-moving events, earnings reports, regulatory changes, and general sentiment.
    3. Summarize the key headlines and the overall sentiment (Positive, Negative, Neutral).
    
    Be concise and cite sources if available in the search results.
    """
    
    # Create the agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    tickers = state["tickers"]
    
    # Invoke the agent
    result = agent.invoke({"messages": [("human", f"Find and analyze news for the following tickers: {tickers}")]})
    
    # The result contains the full state of the agent, including messages.
    last_message = result["messages"][-1]
    
    return {"news_analysis": last_message.content}
