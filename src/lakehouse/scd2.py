"""Slowly Changing Dimension Type 2 merge, written in pure DataFrame operations.

The same logic maps 1:1 onto a Delta Lake `MERGE INTO` (see `scd2_delta_merge_sql`),
but keeping it engine-agnostic makes it fast to unit-test without a Delta runtime.

Dimension columns added/maintained:
    surrogate_key, effective_from, effective_to, is_current, row_hash
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

HIGH_DATE = "9999-12-31 00:00:00"


def _hash(cols: list[str]):
    return F.sha2(F.concat_ws("||", *[F.coalesce(F.col(c).cast("string"), F.lit("∅")) for c in cols]), 256)


def apply_scd2(
    current_dim: DataFrame | None,
    incoming: DataFrame,
    business_key: str,
    tracked_cols: list[str],
    effective_ts_col: str = "updated_at",
) -> DataFrame:
    """Return the full new dimension after applying `incoming` changes.

    - New keys            -> inserted as current
    - Changed tracked cols -> old version closed (effective_to = change ts), new version inserted
    - Unchanged           -> left as-is
    - Multiple changes for one key in a batch are applied in timestamp order.
    """
    attrs = [business_key, *tracked_cols]
    inc = (
        incoming.select(*attrs, effective_ts_col)
        .withColumn("row_hash", _hash(tracked_cols))
        .withColumn("effective_from", F.col(effective_ts_col).cast("timestamp"))
        .drop(effective_ts_col)
    )
    # Collapse consecutive identical versions inside the batch.
    w = Window.partitionBy(business_key).orderBy("effective_from")
    inc = inc.withColumn("_prev_hash", F.lag("row_hash").over(w)).filter(
        F.col("_prev_hash").isNull() | (F.col("_prev_hash") != F.col("row_hash"))
    ).drop("_prev_hash")

    if current_dim is None:
        history_closed = None
        new_versions = inc
    else:
        cur = current_dim.filter("is_current")
        hist = current_dim.filter(~F.col("is_current"))

        # Drop the first incoming version if it equals the current version.
        first_inc = inc.withColumn("_rn", F.row_number().over(w))
        joined = first_inc.join(
            cur.select(business_key, F.col("row_hash").alias("_cur_hash")), business_key, "left"
        )
        same_as_current = (F.col("_rn") == 1) & F.col("_cur_hash").eqNullSafe(F.col("row_hash"))
        new_versions = joined.filter(~same_as_current).drop("_rn", "_cur_hash")

        changed_keys = new_versions.select(business_key).distinct()
        cur_unchanged = cur.join(changed_keys, business_key, "left_anti")
        cur_to_close = cur.join(changed_keys, business_key, "left_semi").drop("effective_to", "is_current")
        history_closed = hist.unionByName(cur_unchanged)
        new_versions = cur_to_close.drop("surrogate_key").unionByName(new_versions)

    w2 = Window.partitionBy(business_key).orderBy("effective_from")
    rebuilt = (
        new_versions.withColumn("_next_from", F.lead("effective_from").over(w2))
        .withColumn("effective_to", F.coalesce(F.col("_next_from"), F.to_timestamp(F.lit(HIGH_DATE))))
        .withColumn("is_current", F.col("_next_from").isNull())
        .drop("_next_from")
        .withColumn("surrogate_key", _hash([business_key, "effective_from"]).substr(1, 16))
    )

    cols = ["surrogate_key", *attrs, "row_hash", "effective_from", "effective_to", "is_current"]
    rebuilt = rebuilt.select(*cols)
    return rebuilt if history_closed is None else history_closed.select(*cols).unionByName(rebuilt)


def scd2_delta_merge_sql(target: str, source_view: str, key: str, tracked: list[str]) -> str:
    """Equivalent Delta Lake MERGE for Databricks (staged-updates pattern)."""
    change = " OR ".join(f"NOT (t.{c} <=> s.{c})" for c in tracked)
    cols = ", ".join([key, *tracked])
    vals = ", ".join(f"s.{c}" for c in [key, *tracked])
    return f"""
MERGE INTO {target} t
USING (
  SELECT s.{key} AS merge_key, s.* FROM {source_view} s
  UNION ALL
  SELECT NULL AS merge_key, s.* FROM {source_view} s
  JOIN {target} t ON s.{key} = t.{key} AND t.is_current
  WHERE {change}
) s
ON t.{key} = s.merge_key AND t.is_current
WHEN MATCHED AND ({change}) THEN
  UPDATE SET t.is_current = false, t.effective_to = s.updated_at
WHEN NOT MATCHED THEN
  INSERT ({cols}, effective_from, effective_to, is_current)
  VALUES ({vals}, s.updated_at, TIMESTAMP '{HIGH_DATE}', true)
""".strip()
