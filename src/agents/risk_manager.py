from langchain_core.prompts import ChatPromptTemplate
from ..state import AgentState
from ..utils import get_llm

def risk_manager_node(state: AgentState):
    """
    Risk Manager that assesses risks based on data and news analysis.
    """
    llm = get_llm(temperature=0)
    
    system_prompt = """You are a Risk Manager for an investment firm.
    Your job is to assess the downside risks, volatility, and potential tail events for the tickers based on the provided analysis.
    
    Input:
    - Data Analysis (Price trends, fundamentals)
    - News Analysis (Sentiment, events)
    
    Output:
    - Identify 3-5 key risk factors (e.g., high volatility, negative sentiment, regulatory headwinds).
    - Provide a "Risk Score" (Low, Medium, High) with justification.
    - Highlight any "Red Flags" that require immediate attention.
    
    Be conservative and critical.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """Data Analysis:
{data_analysis}

News Analysis:
{news_analysis}

Please provide your risk assessment.""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "data_analysis": state.get("data_analysis", "No data analysis provided."),
        "news_analysis": state.get("news_analysis", "No news analysis provided.")
    })
    
    return {"risk_assessment": response.content}
