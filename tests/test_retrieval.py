import os
from agent.rag.retrieval import retrieve_chunks


def test_retrieve_product_policy_first(project_root: str):
    docs_dir = os.path.join(project_root, "docs")
    res = retrieve_chunks("According to the product policy beverages", top_k=3, docs_path=docs_dir)
    assert isinstance(res, list)
    assert any(str(x.get("id", "")).startswith("product_policy::") for x in res)


def test_retrieve_basic(project_root: str):
    docs_dir = os.path.join(project_root, "docs")
    res = retrieve_chunks("AOV definition", top_k=3, docs_path=docs_dir)
    assert isinstance(res, list)
    assert all("content" in x for x in res)

