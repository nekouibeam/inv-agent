from langgraph.graph import StateGraph, END
from .state import AgentState
from .agents.router import router_node
from .agents.data_analyst import data_analyst_node
from .agents.news_analyst import news_analyst_node
from .agents.risk_manager import risk_manager_node
from .agents.editor import editor_node

def create_graph():
    """
    Creates the Multi-Agent Investment Research Graph.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("data_analyst", data_analyst_node)
    workflow.add_node("news_analyst", news_analyst_node)
    workflow.add_node("risk_manager", risk_manager_node)
    workflow.add_node("editor", editor_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    # Router -> Data Analyst AND News Analyst (Parallel)
    workflow.add_edge("router", "data_analyst")
    workflow.add_edge("router", "news_analyst")

    # Data Analyst -> Risk Manager
    workflow.add_edge("data_analyst", "risk_manager")
    
    # News Analyst -> Risk Manager
    workflow.add_edge("news_analyst", "risk_manager")

    # Risk Manager -> Editor
    workflow.add_edge("risk_manager", "editor")

    # Editor -> End
    workflow.add_edge("editor", END)

    return workflow.compile()
