
# TEST FOR THE EXTRACT TRANSACTION DETAILS NODE

import os
import json # to parse the LLM output
from typing import List, TypedDict #typedict for langgraph state
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
        
# Define the State schema
class State(TypedDict):
    company_name: str
    amount_paid: float
    product_name: str
    num_units: int
    function_call_success: bool
    error_message: str
    
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




# Test the transaction details node directly
if __name__ == "__main__":
    test_cases = [
        "Amazon paid $15,499.99 for 25 units of Cloud Servers",
        "Microsoft purchased 100 licenses of Office 365 for $50,000",
        "Invalid text that won't parse correctly"
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n=== TEST CASE {i} ===")
        print(f"Input: {text}")
        
        test_state = {
            "text": text,
            "company_name": "",
            "amount_paid": 0.0,
            "product_name": "",
            "num_units": 0,
            "function_call_success": False,
            "error_message": ""
        }

        result_state = node_extract_transaction_details(test_state)
        
        if result_state["function_call_success"]:
            print("\nExtracted Details:")
            print(f"Company: {result_state['company_name']}")
            print(f"Amount Paid: ${result_state['amount_paid']:,.2f}")
            print(f"Product: {result_state['product_name']}")
            print(f"Units: {result_state['num_units']}")
            print("\nRaw JSON Output:")
            print(json.dumps({
                "company_name": result_state['company_name'],
                "amount_paid": result_state['amount_paid'],
                "product_name": result_state['product_name'],
                "num_units": result_state['num_units']
            }, indent=2))
        else:
            print(f"\nError: {result_state['error_message']}")
