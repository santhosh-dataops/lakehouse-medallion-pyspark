"""End-to-end medallion run on local Parquet (swap format to "delta" on Databricks).

    PYTHONPATH=src python -m lakehouse.pipeline --lake ./lake --batches 3
"""
import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from pyspark.sql import SparkSession

from lakehouse.layers import gold_customer_rfm, gold_daily_sales, silver_customers, silver_sales, to_bronze
from lakehouse.scd2 import apply_scd2

TIERS = ["BRONZE", "SILVER", "GOLD", "PLATINUM"]
STATES = ["TX", "CA", "NY", "FL", "WA"]


def make_batch(landing: Path, batch_no: int, n_customers=300, n_orders=2000, seed=11):
    rnd = random.Random(seed + batch_no)
    day0 = datetime(2026, 9, 1) + timedelta(days=batch_no * 7)
    cust_dir, sales_dir = landing / f"customers/batch={batch_no}", landing / f"sales/batch={batch_no}"
    cust_dir.mkdir(parents=True, exist_ok=True)
    sales_dir.mkdir(parents=True, exist_ok=True)

    with (cust_dir / "customers.json").open("w") as f:
        for i in range(1, n_customers + 1):
            if batch_no and rnd.random() > 0.15:  # only ~15% of customers change per batch
                continue
            f.write(json.dumps({
                "customer_id": f"C{i:05d}", "full_name": f"customer {i}", "email": f"C{i}@Example.com ",
                "state": rnd.choice(STATES), "loyalty_tier": TIERS[min(3, batch_no + rnd.randint(0, 1))],
                "updated_at": (day0 - timedelta(hours=1)).isoformat(),
            }) + "\n")

    with (sales_dir / "sales.json").open("w") as f:
        for _ in range(n_orders):
            oid = f"O{uuid.UUID(int=rnd.getrandbits(128)).hex[:10]}"
            for line in range(rnd.randint(1, 3)):
                rec = {
                    "order_id": oid, "line_id": str(line), "customer_id": f"C{rnd.randint(1, n_customers):05d}",
                    "sku": f"sku-{rnd.randint(1, 400)}", "channel": rnd.choice(["web", "store", "mobile"]),
                    "quantity": rnd.randint(1, 4), "unit_price": round(rnd.uniform(3, 250), 2),
                    "discount_pct": rnd.choice([None, 0.0, 0.1, 0.2]),
                    "order_ts": (day0 + timedelta(minutes=rnd.randint(0, 7 * 24 * 60))).isoformat(),
                }
                if rnd.random() < 0.01:
                    rec["quantity"] = -5  # dirty record -> quarantine
                f.write(json.dumps(rec) + "\n")


def run(lake: str, batches: int) -> None:
    spark = SparkSession.builder.master("local[*]").appName("medallion").config(
        "spark.sql.session.timeZone", "UTC").getOrCreate()
    root = Path(lake)
    dim = None

    for b in range(batches):
        make_batch(root / "landing", b)
        bid = f"b{b:03d}"

        b_cust = to_bronze(spark.read.json(str(root / f"landing/customers/batch={b}")), "crm", bid)
        b_sales = to_bronze(spark.read.json(str(root / f"landing/sales/batch={b}")), "pos", bid)
        b_cust.write.mode("append").parquet(str(root / "bronze/customers"))
        b_sales.write.mode("append").parquet(str(root / "bronze/sales"))

        dim = apply_scd2(dim, silver_customers(b_cust), "customer_id",
                         ["full_name", "email", "state", "loyalty_tier"]).localCheckpoint()
        print(f"[{bid}] dim_customer rows={dim.count()} current={dim.filter('is_current').count()}")

    dim.write.mode("overwrite").parquet(str(root / "silver/dim_customer"))
    sales, quarantined, report = silver_sales(spark.read.parquet(str(root / "bronze/sales")))
    sales.write.mode("overwrite").partitionBy("order_date").parquet(str(root / "silver/sales"))
    quarantined.write.mode("overwrite").parquet(str(root / "quarantine/sales"))
    print(f"silver sales={sales.count()} quarantined={quarantined.count()} dq={report}")

    daily = gold_daily_sales(sales, dim)
    daily.write.mode("overwrite").parquet(str(root / "gold/daily_sales"))
    rfm = gold_customer_rfm(sales, as_of="2026-10-01")
    rfm.write.mode("overwrite").parquet(str(root / "gold/customer_rfm"))
    rfm.groupBy("segment").count().orderBy("segment").show()
    daily.orderBy("order_date", "channel", "loyalty_tier").show(8)
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lake", default="./lake")
    p.add_argument("--batches", type=int, default=3)
    a = p.parse_args()
    run(a.lake, a.batches)
