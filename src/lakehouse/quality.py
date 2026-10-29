"""Lightweight, declarative data-quality expectations (Great Expectations-style).

Each expectation is a named boolean column expression. `apply_expectations`
splits rows into passing and quarantined frames and returns per-rule failure
counts so the run can be logged, alerted on, or failed above a threshold.
"""
from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class Expectation:
    name: str
    rule: Column
    severity: str = "quarantine"  # "quarantine" drops the row; "warn" only counts it


def not_null(col: str) -> Expectation:
    return Expectation(f"{col}_not_null", F.col(col).isNotNull())


def in_set(col: str, values: list) -> Expectation:
    return Expectation(f"{col}_in_set", F.col(col).isin(values))


def _safe(v) -> str:
    return str(v).replace(".", "p").replace("-", "neg")


def between(col: str, lo, hi, severity="quarantine") -> Expectation:
    return Expectation(f"{col}_between_{_safe(lo)}_{_safe(hi)}", F.col(col).between(lo, hi), severity)


def matches(col: str, pattern: str, severity="quarantine") -> Expectation:
    return Expectation(f"{col}_matches", F.col(col).rlike(pattern), severity)


class DataQualityError(Exception):
    pass


def apply_expectations(
    df: DataFrame, expectations: list[Expectation], max_quarantine_ratio: float = 0.05
) -> tuple[DataFrame, DataFrame, dict[str, int]]:
    flagged = df
    for e in expectations:
        flagged = flagged.withColumn(f"_dq_{e.name}", F.coalesce(e.rule, F.lit(False)))

    blocking = [f"_dq_{e.name}" for e in expectations if e.severity == "quarantine"]
    passed_expr = F.lit(True)
    for c in blocking:
        passed_expr = passed_expr & F.col(c)

    failed_names = F.array_compact(
        F.array(*[F.when(~F.col(f"_dq_{e.name}"), F.lit(e.name)) for e in expectations])
    )
    flagged = flagged.withColumn("_dq_passed", passed_expr).withColumn("_dq_failed_rules", failed_names)

    counts_row = flagged.agg(
        F.count("*").alias("_total"),
        *[F.sum(F.when(~F.col(f"_dq_{e.name}"), 1).otherwise(0)).alias(e.name) for e in expectations],
    ).collect()[0]
    total = counts_row["_total"]
    report = {e.name: int(counts_row[e.name] or 0) for e in expectations}

    dq_cols = [c for c in flagged.columns if c.startswith("_dq_") and c != "_dq_failed_rules"]
    good = flagged.filter("_dq_passed").drop(*dq_cols, "_dq_failed_rules")
    bad = flagged.filter(~F.col("_dq_passed")).drop(*dq_cols)

    quarantined = total - good.count()
    if total and quarantined / total > max_quarantine_ratio:
        raise DataQualityError(
            f"{quarantined}/{total} rows quarantined (> {max_quarantine_ratio:.0%}). Failures: {report}"
        )
    return good, bad, report
