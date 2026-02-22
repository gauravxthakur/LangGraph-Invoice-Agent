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

#--------------------------------------------------------------------------------------------

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

# Execute the LangGraph flow
async def run_graph_flow(graph, initial_state: AgentState):
    final_state = await graph.ainvoke(initial_state)
    return final_state

async def chat_interface(graph):
    # Initialize database
    await setup_database()
    
    print("\n================================================")
    print(" Transaction Details Extractor ")
    print("================================================")
    print("Example: 'Amazon paid $40000 for 5 GPUs'")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ('exit', 'quit'):
            break
            
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
        final_state = await run_graph_flow(graph, initial_state)
        
        # 3. Process and display the final result
        print("\n--- Final Result ---")
        if final_state["function_call_success"] and final_state["invoice_success"]:
            print("SUCCESS: Transaction recorded.")
            print(f"   Invoice ID: {final_state['invoice_id']}")
            print(f"   Company: {final_state['company_name']}")
            print(f"   Amount: ${final_state['amount_paid']:,.2f}")
            print(f"   Product: {final_state['product_name']} ({final_state['num_units']} units)")
            display_ledger()
        elif not final_state["function_call_success"]:
             print(f"EXTRACTION FAILED: {final_state['error_message']}")
        elif not final_state["invoice_success"]:
             print(f"DATABASE FAILED: {final_state['error_message']}")
        
        print("\n")

# Create a graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("extract_transaction_details", node_extract_transaction_details)
workflow.add_node("create_invoice", node_create_invoice)
workflow.add_node("END", lambda x: x)

# Add edges to the graph
workflow.set_entry_point("extract_transaction_details")
workflow.add_edge("extract_transaction_details", "create_invoice")
workflow.add_edge("create_invoice", "END")

# Compile the graph
graph = workflow.compile()

# Draw the graph
try:
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path="graph.png")
except Exception:
    pass

# Run the system
if __name__ == "__main__":
    asyncio.run(chat_interface(graph))
