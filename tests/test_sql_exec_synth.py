from agent.tools.sqlite_tool import SQLiteTool
from agent.haystack_modules import Synthesizer


def test_sql_exec_and_synth_types_int(db_path: str):
    tool = SQLiteTool(db_path)
    cols_rows = tool.execute("SELECT COUNT(*) FROM Customers;")
    synth = Synthesizer()
    out = synth.synthesize(
        question="How many customers? Return an integer.",
        docs=[],
        sql_result=cols_rows,
        sql="SELECT COUNT(*) FROM Customers;",
        constraints={},
        format_hint="int",
    )
    assert isinstance(out["final_answer"], int)
    assert "Customers" in out["citations"]


def test_sql_exec_and_synth_types_object(db_path: str):
    tool = SQLiteTool(db_path)
    sql = (
        """
        SELECT c.CompanyName AS customer,
               SUM((od.UnitPrice - 0.7 * od.UnitPrice) * od.Quantity * (1 - od.Discount)) AS margin
        FROM "Order Details" od
        JOIN Orders o ON o.OrderID = od.OrderID
        JOIN Customers c ON c.CustomerID = o.CustomerID
        WHERE date(o.OrderDate) BETWEEN '1997-01-01' AND '1997-12-31'
        GROUP BY c.CompanyName
        ORDER BY margin DESC
        LIMIT 1
        """
    )
    cols_rows = tool.execute(sql)
    synth = Synthesizer()
    out = synth.synthesize(
        question="Per KPI, who was the top customer by gross margin in 1997?",
        docs=[],
        sql_result=cols_rows,
        sql=sql,
        constraints={"year": "1997"},
        format_hint="{customer:str, margin:float}",
    )
    assert isinstance(out["final_answer"], dict)
    assert set(out["final_answer"].keys()) == {"customer", "margin"}
    assert "Orders" in out["citations"] and "Order Details" in out["citations"] and "Customers" in out["citations"]
