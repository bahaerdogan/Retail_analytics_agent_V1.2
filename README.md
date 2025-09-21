Agentic Retail Analyst (Haystack + LangGraph)

A production-ready local agent for retail analytics over the Northwind SQLite database and associated documentation corpus. The system integrates TF-IDF document retrieval, template-based NL→SQL generation with repair mechanisms, and typed output synthesis with citations. It is fully orchestrated using a 7-node LangGraph pipeline, operating entirely offline with no external API calls.

Architecture

graph_hybrid.py: LangGraph workflow (router → retriever → planner → nl2sql → executor → repair → synthesizer)

haystack_modules.py: Core logic for routing, planning, SQL generation, and synthesis

rag/retrieval.py: TF-IDF-based document retrieval with scoring and chunking

tools/sqlite_tool.py: Database schema introspection and query execution

Key Features

Intelligent Repair: Automatically relaxes date filters when queries return no results

Multi-Attempt Strategy: Falls back from exact date ranges to month-day patterns across years

Robust Error Handling: Graceful degradation with fallback strategies

Complete Traceability: JSONL event logging for full auditability

How to Run
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Prepare database
mkdir -p data
curl -L -o data/northwind.sqlite \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# Run hybrid agent on sample batch
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl


Outputs: outputs_hybrid.jsonl (6 lines of synthesized answers)

Trace Logs: agent_trace.log (JSONL events per pipeline node)

Technical Details

Output Compliance: Responses strictly follow format_hint specifications with typed outputs

Citation System: References database tables and document chunks for transparency

Performance: Optimized for local execution with efficient caching and minimal memory use

Extensibility: Modular design allows component replacement while preserving interfaces