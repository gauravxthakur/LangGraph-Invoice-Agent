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
        input_variables=["text"],
        template=(
            "Extract these fields from the input text as STRICT JSON (no commentary):\n"
            "- company_name (string)\n"
            "- amount_paid (float)\n"
            "- product_name (string)\n"
            "- num_units (integer)\n\n"
            "Text: {text}"
        ),
    )
    message = HumanMessage(content=prompt.format(text=text))

    try:
        response = await llm.ainvoke([message])
        result = str(response.content).strip()

        # Handle fenced code blocks like ```json ... ``` or ``` ... ```
        if result.startswith("```"):
            parts = result.split("\n", 1)
            result = parts[1] if len(parts) > 1 else ""
            result = result.rsplit("\n", 1)[0] if "\n" in result else result

        data = json.loads(result)

        company_name = str(data.get("company_name", "")).strip()
        product_name = str(data.get("product_name", "")).strip()

        amount_raw = data.get("amount_paid", 0.0)
        num_units_raw = data.get("num_units", 0)

        amount_paid = float(amount_raw)
        num_units = int(num_units_raw)

        if not company_name:
            raise ValueError("company_name is missing")

        return {
            "company_name": company_name,
            "amount_paid": amount_paid,
            "product_name": product_name,
            "num_units": num_units,
            "success": True,
            "function_call_success": True,
            "error_message": None,
        }
    except json.JSONDecodeError as e:
        return {
            "company_name": "",
            "amount_paid": 0.0,
            "product_name": "",
            "num_units": 0,
            "success": False,
            "function_call_success": False,
            "error_message": f"Invalid JSON from model: {e}",
        }
    except (ValueError, TypeError) as e:
        return {
            "company_name": "",
            "amount_paid": 0.0,
            "product_name": "",
            "num_units": 0,
            "success": False,
            "function_call_success": False,
            "error_message": f"Data validation error: {e}",
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

        print(ledger_str)
        
        return ledger_str
    except sqlite3.Error as e:
        return f"Error displaying ledger: {e}"
    finally:
        if 'conn' in locals() and conn:
            conn.close()