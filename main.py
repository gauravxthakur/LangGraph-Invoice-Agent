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
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools import DATABASE_FILE, setup_database, extract_transaction_details, create_invoice, get_ledger_data
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

load_dotenv()

        
# -----------------------------------STATE SCHEMA-------------------------------------------
class AgentState(TypedDict):
    
    # Conversation History
    messages: Annotated[list[AnyMessage], add_messages]
    
    # Current Input
    text: str
    
    # Extraction output
    company_name: str
    amount_paid: float
    product_name: str
    num_units: int
    
    # Status
    function_call_success: Optional[bool]
    error_message: Optional[str]
    
    # Database output
    invoice_id: Optional[int]
    invoice_success: Optional[bool]
    
#--------------------------------------------------------------------------------------------


# Initialise the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)



#-------------------------------------TOOLS---------------------------------------------
local_tools = [
    extract_transaction_details,
    create_invoice,
    get_ledger_data
]


#------------------------------------AI ASSISTANT---------------------------------------
async def assistant(state: AgentState):
    
    sys_msg = SystemMessage(content=f"""
    You are an intelligent Invoice Processing Assistant that helps users extract transaction details
    from natural language and store them in a database.

    Your Workflow:
    1. When user provides transaction text, use extract_transaction_details() to parse it
    2. If extraction succeeds, use create_invoice() to store the data
    3. If user asks to see records, use get_ledger_data() to display them
    4. Always handle errors gracefully and inform users of the result

    Important Rules:
    - Extract EXACT values from user text (don't make up data)
    - Handle currency symbols and numbers correctly
    - Return clear success/failure messages
    - Use proper error handling for invalid data

    Example 1:
    User: "Amazon paid $40000 for 5 GPUs"
    → extract_transaction_details("Amazon paid $40000 for 5 GPUs")
    → create_invoice("Amazon", 40000.0, "GPUs", 5)
    → "Successfully created invoice #123 for Amazon - $40,000 for 5 GPUs"

    Example 2:
    User: "Microsoft bought 10 laptops for $15000"
    → extract_transaction_details("Microsoft bought 10 laptops for $15000")
    → create_invoice("Microsoft", 15000.0, "laptops", 10)
    → "Successfully created invoice #124 for Microsoft - $15,000 for 10 laptops"

    Example 3:
    User: "Show me all transactions"
    → get_ledger_data()
    → [Display formatted ledger table]
    
    """)
    
    llm_with_tools = llm.bind_tools(local_tools)
    return{
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "text": state["text"],
        "company_name": state["company_name"],
        "amount_paid": state["amount_paid"],
        "product_name": state["product_name"],
        "num_units": state["num_units"],
        "function_call_success": state["function_call_success"],
        "error_message": state["error_message"],
        "invoice_id": state["invoice_id"],
        "invoice_success": state["invoice_success"]
        
    }

#--------------------------------Build the Graph -----------------------------------------------------
async def build_graph():
    "Build the state graph with peoperly initialized tools and assistant function"
    builder = StateGraph(AgentState)
    
    #----------------Nodes-------------------------
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(local_tools)) # ToolNode is the built-in automatic tool execution manager
    
    #----------------Edges-------------------------
    builder.add_edge(START, "assistant") # Start with AI assistant
    builder.add_conditional_edges(
        "assistant",
        tools_condition # Built-in LangGraph condition
    )
    builder.add_edge("tools", "assistant") # After tools are executed, go back to assistant
    
    
    # Initialize Redis checkpointer
    async with AsyncRedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
        app = builder.compile(checkpointer=checkpointer) # Compile the graph
    
    # Generate PNG image of the graph
    image_data = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(image_data)
        
    return app



#----------------------------Additional Functions--------------------------------------

def display_ledger():
    """Print ledger table to console"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Ledger ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        
        print("\n=== Current Ledger ===")
        print("ID | Company Name        | Amount Paid | Product   | Units | Timestamp")
        print("-" * 75)
        for row in rows:
            print(f"{row[0]:2} | {row[1]:<18} | ${row[2]:>9,.2f} | {row[3]:<8} | {row[4]:>5} | {row[5]}")
        print("-" * 75)
    except sqlite3.Error as e:
        print(f"Error displaying ledger: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()



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
            "messages": [HumanMessage(content=user_input)],
            "text": user_input,
            "company_name": "",
            "amount_paid": 0.0,
            "product_name": "",
            "num_units": 0,
            "function_call_success": None,
            "error_message": None,
            "invoice_id": None,
            "invoice_success": None,
        }
        
        # 2. Execute the entire LangGraph workflow
        final_state = await graph.ainvoke(initial_state, config=config)
        
        # 3. Process and display the final result
        print("\n--- Final Result ---")
        print(final_state["messages"][-1].content)
        
        print("\n")
        
        
        
#----------------------------------------RUN----------------------------------------------    
if __name__ == "__main__":
    async def main():
        graph = await build_graph()
        await chat_interface(graph)
    asyncio.run(main())