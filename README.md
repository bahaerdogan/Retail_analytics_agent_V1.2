# Retail Analytics Copilot (Hybrid RAG + SQL, LangGraph)

## Overview
Production-ready local agent for retail analytics over Northwind SQLite database and documentation corpus. Features:
- TF-IDF document retrieval
- Template-based NL→SQL generation with intelligent repair
- Typed output synthesis with citations and confidence scoring
- LangGraph orchestration with 7-node pipeline
- Complete offline operation with no external API calls

## Architecture
- **graph_hybrid.py**: LangGraph workflow with router → retriever → planner → nl2sql → executor → repair → synthesizer
- **haystack_modules.py**: Core business logic modules for routing, planning, SQL generation, and synthesis
- **rag/retrieval.py**: TF-IDF-based document retrieval with chunk scoring
- **tools/sqlite_tool.py**: Database schema introspection and query execution

## Key Features
- **Intelligent Repair**: Automatically relaxes date filters when queries return empty results
- **Multi-attempt Strategy**: Falls back from precise date ranges to month-day patterns across years
- **Robust Error Handling**: Graceful degradation with meaningful fallbacks
- **Complete Traceability**: JSONL event logging for full audit trail

## How to run
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Ensure DB exists
 mkdir -p data
 curl -L -o data/northwind.sqlite \
   https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl
```

- Outputs: `outputs_hybrid.jsonl` (6 lines)
- Trace: `agent_trace.log` (JSONL events per node)

## Technical Details
- **Output Compliance**: All responses match exact format_hint specifications with proper typing
- **Citation System**: Comprehensive referencing of database tables and document chunks
- **Performance**: Optimized for local execution with efficient caching and minimal memory footprint
- **Extensibility**: Modular design allows easy replacement of components while maintaining interface contracts
