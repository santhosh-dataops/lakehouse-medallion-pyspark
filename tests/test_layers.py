import pytest
from pyspark.sql import functions as F

from lakehouse import quality as dq
from lakehouse.layers import gold_daily_sales, silver_sales
from lakehouse.scd2 import apply_scd2

SALES = (
    "order_id string, line_id string, customer_id string, sku string, channel string, quantity long, "
    "unit_price double, discount_pct double, order_ts string, _ingested_at timestamp, _batch_id string"
)


def sale(oid, line, cust, qty, price, ts, batch="b0", ingested="2026-01-01 00:00:00", channel="web", disc=None):
    from datetime import datetime
    return (oid, line, cust, "sku-1", channel, qty, price, disc, ts, datetime.fromisoformat(ingested), batch)


def test_silver_dedups_resent_lines_keeping_latest(spark):
    df = spark.createDataFrame([
        sale("o1", "0", "c1", 1, 10.0, "2026-01-05T10:00:00", "b0", "2026-01-05 00:00:00"),
        sale("o1", "0", "c1", 2, 10.0, "2026-01-05T10:00:00", "b1", "2026-01-06 00:00:00"),
    ], SALES)
    out, _, _ = silver_sales(df, max_quarantine_ratio=1.0)
    rows = out.collect()
    assert len(rows) == 1 and rows[0].quantity == 2


def test_silver_quarantines_bad_rows_and_computes_net(spark):
    df = spark.createDataFrame([
        sale("o1", "0", "c1", 2, 50.0, "2026-01-05T10:00:00", disc=0.1),
        sale("o2", "0", "c1", -3, 50.0, "2026-01-05T10:00:00"),
        sale("o3", "0", None, 1, 5.0, "2026-01-05T10:00:00"),
    ], SALES)
    good, bad, report = silver_sales(df, max_quarantine_ratio=1.0)
    assert good.collect()[0].net_amount == pytest.approx(90.0)
    assert bad.count() == 2
    assert report["quantity_between_1_1000"] == 1 and report["customer_id_not_null"] == 1


def test_quality_gate_fails_run_above_threshold(spark):
    df = spark.createDataFrame([(None,), (None,), ("x",)], "id string")
    with pytest.raises(dq.DataQualityError):
        dq.apply_expectations(df, [dq.not_null("id")], max_quarantine_ratio=0.2)


def test_warn_severity_does_not_quarantine(spark):
    df = spark.createDataFrame([(5,), (500,)], "v int")
    good, bad, report = dq.apply_expectations(df, [dq.between("v", 0, 10, severity="warn")])
    assert good.count() == 2 and report["v_between_0_10"] == 1


def test_gold_uses_point_in_time_customer_version(spark):
    cust_schema = "customer_id string, state string, loyalty_tier string, updated_at string"
    dim = apply_scd2(None, spark.createDataFrame([
        ("c1", "TX", "SILVER", "2026-01-01T00:00:00"),
        ("c1", "TX", "GOLD", "2026-01-10T00:00:00"),
    ], cust_schema), "customer_id", ["state", "loyalty_tier"])

    df = spark.createDataFrame([
        sale("o1", "0", "c1", 1, 100.0, "2026-01-05T12:00:00"),  # SILVER era
        sale("o2", "0", "c1", 1, 40.0, "2026-01-15T12:00:00"),   # GOLD era
    ], SALES)
    sales, _, _ = silver_sales(df, max_quarantine_ratio=1.0)
    out = {r.loyalty_tier: r.net_revenue for r in gold_daily_sales(sales, dim).collect()}
    assert out == {"SILVER": 100.0, "GOLD": 40.0}
