## Tech Stack
Orchestration with **LangGraph** and **Pydantic**

Persistence & Memory with **Redis checkpointers** and **MongoDb (via MCP)**

Evaluation & Reliability with **LangSmith** and **DeepEval**


## GRAPH
![graph image](image.png)


## AGENT
<img width="955" height="870" alt="Screenshot 2026-02-27 043850" src="https://github.com/user-attachments/assets/2e34833c-e511-4410-ab85-108309fc251b" />


## RESOURCES
LangGraph Redis https://github.com/redis-developer/langgraph-redis

MongoDB MCP Server https://github.com/mongodb-js/mongodb-mcp-server

A quick guide on LLM Evaluation Metrics https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation


## How to Install and Run
1. Clone the repository

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Start Redis container
```bash
docker start redis-stack
# Or create if first time:
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

3. Set up environment variables

4. Run the application
```bash
py main.py
```

5. Connect to LangSmith (Optional but Recommended for Observability)
