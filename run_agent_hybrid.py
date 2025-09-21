import json
import click
from agent.graph_hybrid import build_graph, router_node, retriever_node, planner_node, nl2sql_node, executor_node, repair_node, synthesizer_node

@click.command()
@click.option("--batch", required=True, help="Path to JSONL input with questions")
@click.option("--out", required=True, help="Path to JSONL output file")
def main(batch, out):
    graph = build_graph()

    with open(batch, "r") as f:
        questions = [json.loads(line) for line in f]

    outputs = []
    for q in questions:
        state = {"question": q["question"], "id": q["id"], "format_hint": q.get("format_hint", "")}
        result = graph.invoke(state)
        if not result or not isinstance(result, dict) or "final_answer" not in result:
            s = dict(state)
            s = router_node(s)
            s = retriever_node(s)
            s = planner_node(s)
            s = nl2sql_node(s)
            s = executor_node(s)
            s = repair_node(s)
            s = synthesizer_node(s)
            result = s

        outputs.append({
            "id": q["id"],
            "final_answer": result.get("final_answer"),
            "sql": result.get("sql", ""),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", ""),
            "citations": result.get("citations", [])
        })

    with open(out, "w") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")

if __name__ == "__main__":
    main()