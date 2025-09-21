from agent.haystack_modules import NL2SQL, Planner
from agent.tools.sqlite_tool import SQLiteTool


def test_nl2sql_generates_sql():
    schema = SQLiteTool().get_schema()
    cons = Planner().extract("Top 3 products by revenue", [])
    sql = NL2SQL().generate("Top 3 products by revenue all-time", schema=schema, constraints=cons)
    assert isinstance(sql, str)
    assert "SELECT" in sql.upper()


def test_nl2sql_date_range_respected():
    schema = SQLiteTool().get_schema()
    cons = {"date_range": ("1997-12-01", "1997-12-31")}
    sql = NL2SQL().generate("AOV in Winter Classics 1997", schema=schema, constraints=cons)
    assert "1997-12-01" in sql and "1997-12-31" in sql

