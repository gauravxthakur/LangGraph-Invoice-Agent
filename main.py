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


# Database setup
DATABASE_FILE = "ledger_test.db"

def setup_database():
    """Initializes the SQLite database and creates the Ledger table."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Ledger (
            id INTEGER PRIMARY KEY,
            company_name TEXT NOT NULL,
            amount_paid REAL NOT NULL,
            product_name TEXT,
            num_units INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB SETUP] Database '{DATABASE_FILE}' initialized and 'Ledger' table ready.")    
    
# Initialise the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# Extract transaction details
def node_extract_transaction_details(state: State):
    ''' Extract transaction details '''
    prompt = PromptTemplate(
        input_variables = ["text"],
        template = """Extract these fields from {text} as JSON:
- company_name (string) 
- amount_paid (float)
- product_name (string)
- num_units (integer)"""
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    
    try:
        result = llm.invoke([message]).content.strip()
        # Handle both ```json and ``` cases
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("\n", 1)[0]
        data = json.loads(result)
        
        state["company_name"] = str(data.get("company_name", "")).strip()
        state["amount_paid"] = float(data.get("amount_paid", 0.0)) 
        state["product_name"] = str(data.get("product_name", "")).strip()
        state["num_units"] = int(data.get("num_units", 0))
        state["function_call_success"] = True
        
    except json.JSONDecodeError as e:
        state["error_message"] = f"Invalid JSON: {e}"
        state["function_call_success"] = False
    except (KeyError, ValueError) as e: 
        state["error_message"] = f"Data validation error: {e}"
        state["function_call_success"] = False
    
    return state

def node_create_invoice(state: AgentState):
    '''Create invoice and store in database'''
    if not state.get("function_call_success", False):
        state["error_message"] = "Cannot create invoice - extraction failed"
        return state
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO Ledger (company_name, amount_paid, product_name, num_units)
            VALUES (?, ?, ?, ?)
            """,
            (state["company_name"], state["amount_paid"], 
             state["product_name"], state["num_units"])
        )
        conn.commit()
        state["invoice_id"] = cursor.lastrowid
        state["invoice_success"] = True
        
    except sqlite3.Error as e:
        state["error_message"] = f"Database error: {e}"
        state["invoice_success"] = False
        
    finally:
        conn.close()
        
    return state

def display_ledger():
    """Display current ledger table"""
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
def run_graph_flow(graph, initial_state: AgentState):
    final_state = graph.invoke(initial_state)
    return final_state

def chat_interface(graph):
    # Initialize database
    setup_database()
    
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
        initial_state: State = {
            "text": user_input,
            "company_name": "",
            "amount_paid": 0.0,
            "product_name": "",
            "num_units": 0,
            "function_call_success": False,
            "error_message": "",
            "invoice_id": 0,
            "invoice_success": False
        }
        
        # 2. Execute the entire LangGraph workflow
        final_state = run_graph_flow(graph, initial_state)
        
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
    chat_interface(graph)
