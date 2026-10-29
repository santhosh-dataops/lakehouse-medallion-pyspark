# Retail Lakehouse — Medallion Architecture in PySpark

A bronze → silver → gold lakehouse for retail sales and customer data, featuring a tested **SCD Type 2** customer dimension, a declarative **data-quality framework** with quarantine, and **point-in-time** joins in the gold layer.

![CI](https://github.com/GITHUB_USERNAME/lakehouse-medallion-pyspark/actions/workflows/ci.yml/badge.svg)

## Architecture

```
 landing/ (JSON from CRM + POS, weekly batches)
     │  to_bronze(): append-only + _source_system, _batch_id, _ingested_at, _source_file
     ▼
 BRONZE   customers, sales
     │  silver_*(): typing, trimming, dedup of re-sent lines, DQ expectations
     ▼
 SILVER   sales (partitioned by order_date)   dim_customer (SCD2)   quarantine/sales
     │  point-in-time join on effective_from <= order_ts < effective_to
     ▼
 GOLD     daily_sales (date × channel × loyalty tier)   customer_rfm (R/F/M quintiles + segment)
```

## Highlights

**SCD Type 2 (`src/lakehouse/scd2.py`)**
- Hash-based change detection on tracked columns (null-safe).
- Handles new keys, unchanged records (no-op), and **multiple changes for one key within a single batch**, applied in timestamp order.
- Engine-agnostic DataFrame implementation for fast unit tests, plus `scd2_delta_merge_sql()` which renders the equivalent Delta Lake `MERGE INTO` (staged-updates pattern) for Databricks.

**Data quality (`src/lakehouse/quality.py`)**
- Declarative expectations: `not_null`, `in_set`, `between`, `matches`.
- `quarantine` vs `warn` severities; failing rows land in a quarantine table with the list of rules they broke.
- Run-level circuit breaker: the job fails if quarantined rows exceed a threshold (default 5%).

**Gold**
- Daily sales attributed to the customer's loyalty tier **as it was at order time**, not as it is today.
- RFM segmentation (Champion, Recent, Loyal-at-risk, Lapsing, Steady).

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m lakehouse.pipeline --lake ./lake --batches 3
```

```
[b001] dim_customer rows=340 current=300
[b002] dim_customer rows=387 current=300
```

On Databricks, switch `.parquet(...)` writes to `.format("delta")` and use `scd2_delta_merge_sql()` for the dimension.

## Tests

```bash
PYTHONPATH=src pytest -q     # 12 tests: SCD2 edge cases, dedup, DQ gates, point-in-time join
```

## Tech
Python · PySpark · Delta Lake · Data Quality · Dimensional Modeling (SCD2) · PyTest · GitHub Actions
