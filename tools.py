import os
import random
import json
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import List
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
import sqlite3
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Optional

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Database setup
DATABASE_FILE = "ledger_test.db"


#------------------------------------------------------------------------------------------------------------

@tool
def setup_database():
    """
    Initializes the SQLite database and creates the Ledger table for invoice storage.
    
    Creates a table with columns: id, company_name, amount_paid, product_name, 
    num_units, and timestamp. Uses 'ledger_test.db' as the database file.
    
    Returns:
        None: Prints setup confirmation message
    
    Note:
        Table is created with IF NOT EXISTS to prevent errors on repeated calls.
    """
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
    
    
    
#--------------------------------------------------------------------------------------------------------------------
    
@tool
async def extract_transaction_details(text: str) -> dict:
    """
    Extracts transaction details from natural language text using a LLM.
    
    Args:
        text (str): Natural language text to parse
        
    Returns:
        dict: Extracted data with keys: company_name, amount_paid, 
              product_name, num_units, success, error_message
    """

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
        response = await llm.ainvoke([message])
        result = response.content.strip()
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


#-------------------------------------------------------------------------------------------------

@tool
async def create_invoice(company_name: str, amount_paid: float, 
                         product_name: str, num_units: int) -> dict:
    """
    Creates and stores invoice record in SQLite database.
    
    Args:
        company_name (str): Company name
        amount_paid (float): Payment amount
        product_name (str): Product description
        num_units (int): Quantity purchased
        
    Returns:
        dict: {invoice_id: int, success: bool, error_message: str}
    """
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


#----------------------------------------------------------------------------------------------

@tool
def get_ledger_data(): 
    """
    Retrieves and formats all invoice records from the database.
    
    Queries the Ledger table and returns a formatted string containing
    all transaction records sorted by timestamp (newest first).
    
    Returns:
        str: Formatted ledger table with ID, company name, amount,
             product, units, and timestamp for all records
             
    Note:
        Prints formatted table directly to console for user display.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Ledger ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        
        # Build string instead of printing
        ledger_str = "\n=== Current Ledger ===\n"
        ledger_str += "ID | Company Name        | Amount Paid | Product   | Units | Timestamp\n"
        ledger_str += "-" * 75 + "\n"
        for row in rows:
            ledger_str += f"{row[0]:2} | {row[1]:<18} | ${row[2]:>9,.2f} | {row[3]:<8} | {row[4]:>5} | {row[5]}\n"
        ledger_str += "-" * 75 + "\n"
        
        return ledger_str
    except sqlite3.Error as e:
        return f"Error displaying ledger: {e}"
    finally:
        if 'conn' in locals() and conn:
            conn.close()