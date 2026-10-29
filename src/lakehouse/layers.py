"""Bronze -> Silver -> Gold transformations for a retail sales lakehouse."""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from lakehouse import quality as dq


# ---------- Bronze: raw, append-only, with lineage metadata ----------
def to_bronze(raw: DataFrame, source_system: str, batch_id: str) -> DataFrame:
    return (
        raw.withColumn("_source_system", F.lit(source_system))
        .withColumn("_batch_id", F.lit(batch_id))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )


# ---------- Silver: cleansed, conformed, deduplicated ----------
def sales_expectations() -> list[dq.Expectation]:
    return [
        dq.not_null("order_id"),
        dq.not_null("customer_id"),
        dq.not_null("order_ts"),
        dq.between("quantity", 1, 1000),
        dq.between("unit_price", 0.01, 100_000),
        dq.in_set("channel", ["WEB", "STORE", "MOBILE"]),
    ]


def silver_sales(bronze_sales: DataFrame, max_quarantine_ratio: float = 0.05):
    typed = bronze_sales.select(
        F.trim("order_id").alias("order_id"),
        F.trim("line_id").alias("line_id"),
        F.trim("customer_id").alias("customer_id"),
        F.upper(F.trim("sku")).alias("sku"),
        F.upper(F.trim("channel")).alias("channel"),
        F.col("quantity").cast("int").alias("quantity"),
        F.col("unit_price").cast("decimal(12,2)").alias("unit_price"),
        F.col("discount_pct").cast("double").alias("discount_pct"),
        F.to_timestamp("order_ts").alias("order_ts"),
        "_ingested_at",
        "_batch_id",
    )
    # Latest ingested version of each order line wins (handles re-sent files).
    w = Window.partitionBy("order_id", "line_id").orderBy(F.col("_ingested_at").desc(), F.col("_batch_id").desc())
    deduped = typed.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")

    good, quarantined, report = dq.apply_expectations(deduped, sales_expectations(), max_quarantine_ratio)
    enriched = (
        good.withColumn("discount_pct", F.coalesce("discount_pct", F.lit(0.0)))
        .withColumn("net_amount", F.round(F.col("quantity") * F.col("unit_price") * (1 - F.col("discount_pct")), 2))
        .withColumn("order_date", F.to_date("order_ts"))
    )
    return enriched, quarantined, report


def silver_customers(bronze_customers: DataFrame) -> DataFrame:
    return bronze_customers.select(
        F.trim("customer_id").alias("customer_id"),
        F.initcap(F.trim("full_name")).alias("full_name"),
        F.lower(F.trim("email")).alias("email"),
        F.upper(F.trim("state")).alias("state"),
        F.upper(F.trim("loyalty_tier")).alias("loyalty_tier"),
        F.to_timestamp("updated_at").alias("updated_at"),
    ).filter(F.col("customer_id").isNotNull())


# ---------- Gold: business-level aggregates ----------
def gold_daily_sales(sales: DataFrame, dim_customer: DataFrame) -> DataFrame:
    """Point-in-time join: each sale picks the customer version valid at order time."""
    dim = dim_customer.select("customer_id", "state", "loyalty_tier", "effective_from", "effective_to")
    joined = sales.join(
        dim,
        (sales.customer_id == dim.customer_id)
        & (sales.order_ts >= dim.effective_from)
        & (sales.order_ts < dim.effective_to),
        "left",
    ).drop(dim.customer_id)
    return (
        joined.groupBy("order_date", "channel", F.coalesce("loyalty_tier", F.lit("UNKNOWN")).alias("loyalty_tier"))
        .agg(
            F.countDistinct("order_id").alias("orders"),
            F.sum("quantity").alias("units"),
            F.round(F.sum("net_amount"), 2).alias("net_revenue"),
            F.countDistinct("customer_id").alias("unique_customers"),
        )
        .withColumn("avg_order_value", F.round(F.col("net_revenue") / F.col("orders"), 2))
    )


def gold_customer_rfm(sales: DataFrame, as_of: str) -> DataFrame:
    """Recency / Frequency / Monetary scores (1-5 quintiles) per customer."""
    base = sales.groupBy("customer_id").agg(
        F.datediff(F.to_date(F.lit(as_of)), F.max("order_date")).alias("recency_days"),
        F.countDistinct("order_id").alias("frequency"),
        F.round(F.sum("net_amount"), 2).alias("monetary"),
    )
    scored = (
        base.withColumn("r_score", 6 - F.ntile(5).over(Window.orderBy("recency_days")))
        .withColumn("f_score", F.ntile(5).over(Window.orderBy("frequency")))
        .withColumn("m_score", F.ntile(5).over(Window.orderBy("monetary")))
    )
    return scored.withColumn(
        "segment",
        F.when((F.col("r_score") >= 4) & (F.col("f_score") >= 4), "CHAMPION")
        .when(F.col("r_score") >= 4, "RECENT")
        .when(F.col("f_score") >= 4, "LOYAL_AT_RISK")
        .when(F.col("r_score") <= 2, "LAPSING")
        .otherwise("STEADY"),
    )
