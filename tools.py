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
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Database setup
DATABASE_FILE = "ledger_test.db"


#------------------------------------------------------------------------------------------------------------

async def setup_database():
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
    # print(f"[DB SETUP] Database '{DATABASE_FILE}' initialized and 'Ledger' table ready.")    

    

#---------------------------------------------PYDANTIC SCHEMA-------------------------------------------------------
class TransactionDetails(BaseModel):
    company_name: str = Field(description="The name of the company that made the purchase")
    amount_paid: float = Field(gt=0, description="The total amount paid") # gt=0 means "Greater Than 0"
    product_name: str = Field(description="The name of the item or service bought")
    num_units: int = Field(ge=1, description="The quantity of the product purchased") # ge=1 means "Greater Than or Equal To 1"


#--------------------------------------------------------------------------------------------------------------------
    
@tool
async def extract_transaction_details(text: str) -> dict:
    """
    Extracts transaction details from text using structured LLM output.

    Args:
        text (str): Input text containing transaction information.

    Returns:
        dict: Structured transaction data with company_name, amount_paid, 
        product_name, num_units, success status, and error handling.
    """
    
    structured_llm = llm.with_structured_output(TransactionDetails)
    
    try:
        result = await structured_llm.ainvoke(text)
        return {
            **result.model_dump(), # converts the Pydantic object into a clean Python dictionary
            "success": True,
            "function_call_success": True,
            "error_message": None
        }
    except Exception as e:
        # If the AI hallucinates a string where a float should be, 
        # Pydantic catches it here instead of crashing the program.
        return {
            "company_name": "",
            "amount_paid": 0.0,
            "product_name": "",
            "num_units": 0,
            "success": False,
            "function_call_success": False,
            "error_message": f"Extraction failed: {str(e)}"
        }


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
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO Ledger (company_name, amount_paid, product_name, num_units)
            VALUES (?, ?, ?, ?)
            """,
            (company_name, float(amount_paid), product_name, int(num_units))
        )
        conn.commit()

        invoice_id = cursor.lastrowid
        cursor.execute("SELECT timestamp FROM Ledger WHERE id = ?", (invoice_id,))
        ts_row = cursor.fetchone()
        timestamp = ts_row[0] if ts_row else None

        return {
            "invoice_id": invoice_id,
            "success": True,
            "invoice_success": True,
            "error_message": None,
            "timestamp": timestamp,
        }
        
    except sqlite3.Error as e:
        return {
            "invoice_id": None,
            "success": False,
            "invoice_success": False,
            "error_message": f"Database error: {e}",
            "timestamp": None,
        }
    finally:
        if 'conn' in locals() and conn:
            conn.close()


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