from agent.graph_hybrid import build_graph


def test_graph_runs_for_all_samples(project_root: str):
    graph = build_graph()
    samples = [
        {
            "id": "rag_policy_beverages_return_days",
            "question": "According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.",
            "format_hint": "int",
        },
        {
            "id": "hybrid_top_category_qty_summer_1997",
            "question": "During 'Summer Beverages 1997' as defined in the marketing calendar, which product category had the highest total quantity sold? Return {category:str, quantity:int}.",
            "format_hint": "{category:str, quantity:int}",
        },
    ]
    for s in samples:
        out = graph.invoke({"question": s["question"], "id": s["id"], "format_hint": s["format_hint"]})
        assert isinstance(out, dict)
        assert "final_answer" in out

