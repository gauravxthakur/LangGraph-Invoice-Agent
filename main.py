import asyncio
import os
from dotenv import load_dotenv
import json
import sqlite3
from typing import List, TypedDict, Annotated, Optional
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
from tools import TransactionDetails, DATABASE_FILE, setup_database, extract_transaction_details, create_invoice, get_ledger_data
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

load_dotenv()

        
# -----------------------------------STATE SCHEMA-------------------------------------------
class AgentState(TypedDict):
    
    # Conversation History
    messages: Annotated[list[AnyMessage], add_messages]
 
#--------------------------------------------------------------------------------------------


#-------------------------------------TOOLS---------------------------------------------
local_tools = [
    extract_transaction_details,
    create_invoice,
    get_ledger_data
]


# Initialise the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# System Message
sys_msg = SystemMessage(content=f"""
You are an ERP Assistant for invoice processing.

Tools:
- extract_transaction_details(text): Parse transaction from natural language
- create_invoice(company, amount, product, quantity): Store in database
- get_ledger_data(): Show all transactions

Instructions:
- Extract EXACT details from user text (no invented data)
- Handle currency/numbers correctly
- For "show/display" requests, use get_ledger_data()
- Return clear, user-friendly messages

Examples:
- "Amazon paid $40000 for 5 GPUs" → Extract → Create invoice
- "Show all transactions" → Display ledger
""")


#--------------------------------Build the Graph -----------------------------------------------------
async def build_graph(checkpointer):
    "Build the state graph with peoperly initialized tools and assistant function"
    
    builder = StateGraph(AgentState)
    
    # 1. Setup Tools - Use local tools only
    llm_with_tools = llm.bind_tools(local_tools)
    
    #------------------------------------AI ASSISTANT---------------------------------------
    async def assistant(state: AgentState):
        
        response = await llm_with_tools.ainvoke([sys_msg] + state["messages"])
        
        return{
            "messages": [response],
        }

    
    # 2. Nodes & Edges
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(local_tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
           

    app = builder.compile(checkpointer=checkpointer)
    
    # Generate PNG image of the graph
    image_data = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(image_data)
        
    return app



#---------------------------------------CHAT INTERFACE-----------------------------------------
async def chat_interface(graph):
    # Initialize database
    await setup_database()
    
    print("""
          
██╗███╗░░██╗██╗░░░██╗░█████╗░██╗░█████╗░███████╗
██║████╗░██║██║░░░██║██╔══██╗██║██╔══██╗██╔════╝
██║██╔██╗██║╚██╗░██╔╝██║░░██║██║██║░░╚═╝█████╗░░
██║██║╚████║░╚████╔╝░██║░░██║██║██║░░██╗██╔══╝░░
██║██║░╚███║░░╚██╔╝░░╚█████╔╝██║╚█████╔╝███████╗
╚═╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░╚════╝░╚══════╝

░█████╗░░██████╗░██████╗██╗░██████╗████████╗░█████╗░███╗░░██╗████████╗
██╔══██╗██╔════╝██╔════╝██║██╔════╝╚══██╔══╝██╔══██╗████╗░██║╚══██╔══╝
███████║╚█████╗░╚█████╗░██║╚█████╗░░░░██║░░░███████║██╔██╗██║░░░██║░░░
██╔══██║░╚═══██╗░╚═══██╗██║░╚═══██╗░░░██║░░░██╔══██║██║╚████║░░░██║░░░
██║░░██║██████╔╝██████╔╝██║██████╔╝░░░██║░░░██║░░██║██║░╚███║░░░██║░░░
╚═╝░░╚═╝╚═════╝░╚═════╝░╚═╝╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚══╝░░░╚═╝░░░""")
    print("Example: 'Amazon paid $40000 for 5 GPUs'")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ('exit', 'quit'):
            break
        
        # This is what allows Redis to track distinct conversations
        config = {"configurable": {"thread_id": "user_session_001"}}
        
        # 1. Initialize the State object with user input
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        # 2. Execute the entire LangGraph workflow
        final_state = await graph.ainvoke(initial_state, config=config)
        
        # 3. Process and display the final result
        last_message = final_state["messages"][-1]
        content = last_message.content

        if isinstance(content, list):
            # Handle list of content items
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    print(item['text'])
        elif isinstance(content, str):
            # Handle string content
            print(content)
        else:
            # Handle other content types
            print(str(content))
        
        print("\n")
        
    
# ----------------------------------------RUN----------------------------------------------    
async def run_app():
    checkpointer = AsyncRedisSaver.from_conn_string("redis://localhost:6379")
    graph = await build_graph(checkpointer)
    await chat_interface(graph)


if __name__ == "__main__":
    try:
        asyncio.run(run_app())
    except KeyboardInterrupt:
        print("\nSession ended.")