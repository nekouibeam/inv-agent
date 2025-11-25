from langchain_core.prompts import ChatPromptTemplate
from ..state import AgentState
from ..utils import get_llm

def editor_node(state: AgentState):
    """
    Chief Editor that compiles the final investment memo.
    """
    llm = get_llm(temperature=0)
    
    system_prompt = """You are the Chief Editor of a prestigious investment research firm.
    Your goal is to compile a comprehensive Investment Memo based on the reports from your team.
    
    Inputs:
    - Data Analysis
    - News Analysis
    - Risk Assessment
    
    Output:
    - A well-structured Markdown report.
    - Sections:
        1. **Executive Summary**: High-level overview and recommendation (Buy/Hold/Sell - purely opinion based on data).
        2. **Financial Analysis**: Key metrics and trends.
        3. **Market Sentiment & News**: Summary of recent events.
        4. **Risk Factors**: Critical risks to consider.
        5. **Conclusion**: Final verdict.
        
    Tone: Professional, objective, and concise.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """Data Analysis:
{data_analysis}

News Analysis:
{news_analysis}

Risk Assessment:
{risk_assessment}

Please generate the final Investment Memo.""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "data_analysis": state.get("data_analysis"),
        "news_analysis": state.get("news_analysis"),
        "risk_assessment": state.get("risk_assessment")
    })
    
    return {"final_report": response.content}
