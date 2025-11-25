from langchain.agents import create_agent
from ..state import AgentState
from ..tools.finance_tools import get_stock_data
from ..utils import get_llm

def data_analyst_node(state: AgentState):
    """
    Finance Data Analyst that gathers and analyzes market data using a ReAct agent.
    """
    llm = get_llm(temperature=0)
    tools = [get_stock_data]
    
    system_prompt = """You are a Finance Data Analyst. 
    Your goal is to analyze the financial data for the provided tickers.
    
    1. Use the `get_stock_data` tool to fetch data for EACH ticker in the list.
    2. Analyze the price trends (last 1 month), volatility, and key fundamental metrics (PE, Market Cap).
    3. Provide a structured summary of the financial health and recent performance.
    
    Be precise with numbers. If data is missing, state it clearly.
    """
    
    # Create the agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    tickers = state["tickers"]
    
    # Invoke the agent
    # The agent expects a list of messages. We pass the task as a human message.
    result = agent.invoke({"messages": [("human", f"Analyze the following tickers: {tickers}")]})
    
    # The result contains the full state of the agent, including messages.
    # The last message should be the AI's final response.
    last_message = result["messages"][-1]
    
    return {"data_analysis": last_message.content}
