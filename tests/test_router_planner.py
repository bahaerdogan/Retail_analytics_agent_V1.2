from agent.haystack_modules import Router, Planner


def test_router_routes_basic():
    r = Router()
    assert r.classify("According to the product policy") == "rag"
    assert r.classify("Top 3 products by revenue") in {"sql", "hybrid"}
    assert r.classify("AOV in winter 1997") in {"hybrid", "sql"}


def test_planner_dates_from_calendar():
    p = Planner()
    cons = p.extract("During 'Summer Beverages 1997'...", [])
    assert cons.get("date_range") == ("1997-06-01", "1997-06-30")
    cons2 = p.extract("Using the AOV definition during Winter Classics 1997", [])
    assert cons2.get("date_range") == ("1997-12-01", "1997-12-31")

