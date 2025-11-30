from typing import List
from langchain.agents import create_agent
from langchain_core.tools import tool
from ..state import AgentState
from ..utils import get_llm

@tool
def submit_routing_instructions(tickers: List[str], data_analyst_instructions: str, news_analyst_instructions: str):
    """
    Submit the extracted tickers and specific instructions for the Data Analyst and News Analyst.
    
    Args:
        tickers: List of stock tickers found in the query.
        data_analyst_instructions: Specific instructions for the Data Analyst (financials, valuation).
        news_analyst_instructions: Specific instructions for the News Analyst (news, sentiment, events).
    """
    return "Instructions submitted."

def router_node(state: AgentState):
    """
    Router agent that extracts tickers and generates specific instructions for analysts.
    """
    llm = get_llm(temperature=0)
    
    system_prompt = """You are a Senior Financial Research Lead.
    Your job is to orchestrate the research process by analyzing the user's query and delegating tasks.
    
    1. **Analyze the User Query**: Understand the core question, hypothesis, or concern.
    2. **Extract Tickers**: Identify all stock tickers mentioned or implied.
    3. **Delegate to Data Analyst**: Create specific instructions for the Data Analyst. 
       - What specific financial metrics should they look for? (e.g., "Check gross margins if the user asks about profitability").
       - What valuation multiples are relevant?
    4. **Delegate to News Analyst**: Create specific instructions for the News Analyst.
       - What specific keywords or topics should they search for? (e.g., "Search for 'supply chain issues' if the user asks about delays").
       - What sentiment or events matter most?
       
    **Goal**: Do not just pass the generic query. Translate the user's intent into precise, actionable technical instructions for your team.
    
    You MUST call the `submit_routing_instructions` tool to output your decision.
    """
    
    # Create the agent
    agent = create_agent(
        model=llm,
        tools=[submit_routing_instructions],
        system_prompt=system_prompt
    )
    
    # Invoke the agent
    result = agent.invoke({"messages": [("human", state["query"])]})
    
    # Extract the tool call from the last message (or the one before if the last is a tool message)
    # The agent should have called the tool.
    # We need to find the tool call in the messages.
    
    messages = result["messages"]
    tool_call = None
    
    # Iterate backwards to find the tool call
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]
            break
            
    if tool_call and tool_call["name"] == "submit_routing_instructions":
        args = tool_call["args"]
        return {
            "tickers": args.get("tickers", []),
            "data_analyst_instructions": args.get("data_analyst_instructions", ""),
            "news_analyst_instructions": args.get("news_analyst_instructions", "")
        }
    
    # Fallback if no tool call (shouldn't happen with good LLM)
    return {"tickers": [], "data_analyst_instructions": state["query"], "news_analyst_instructions": state["query"]}
