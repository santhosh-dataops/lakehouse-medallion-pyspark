from lakehouse.scd2 import apply_scd2, scd2_delta_merge_sql

SCHEMA = "customer_id string, state string, tier string, updated_at string"
TRACKED = ["state", "tier"]


def dim_of(spark, rows):
    return apply_scd2(None, spark.createDataFrame(rows, SCHEMA), "customer_id", TRACKED)


def versions(df, key):
    return [
        (r.state, r.tier, r.is_current)
        for r in df.filter(f"customer_id = '{key}'").orderBy("effective_from").collect()
    ]


def test_initial_load_marks_all_current(spark):
    dim = dim_of(spark, [("c1", "TX", "GOLD", "2026-01-01"), ("c2", "CA", "SILVER", "2026-01-01")])
    assert dim.count() == 2
    assert dim.filter("is_current").count() == 2


def test_change_closes_old_version_and_opens_new(spark):
    dim = dim_of(spark, [("c1", "TX", "GOLD", "2026-01-01")])
    inc = spark.createDataFrame([("c1", "TX", "PLATINUM", "2026-02-01")], SCHEMA)
    new = apply_scd2(dim, inc, "customer_id", TRACKED)
    assert versions(new, "c1") == [("TX", "GOLD", False), ("TX", "PLATINUM", True)]
    old = new.filter("not is_current").collect()[0]
    assert str(old.effective_to).startswith("2026-02-01")


def test_unchanged_record_is_noop(spark):
    dim = dim_of(spark, [("c1", "TX", "GOLD", "2026-01-01")])
    inc = spark.createDataFrame([("c1", "TX", "GOLD", "2026-03-01")], SCHEMA)
    new = apply_scd2(dim, inc, "customer_id", TRACKED)
    assert new.count() == 1
    assert sorted(r.surrogate_key for r in new.collect()) == sorted(r.surrogate_key for r in dim.collect())


def test_new_key_inserted_and_untouched_keys_kept(spark):
    dim = dim_of(spark, [("c1", "TX", "GOLD", "2026-01-01")])
    inc = spark.createDataFrame([("c2", "NY", "BRONZE", "2026-02-01")], SCHEMA)
    new = apply_scd2(dim, inc, "customer_id", TRACKED)
    assert new.filter("is_current").count() == 2


def test_multiple_changes_in_one_batch_are_ordered(spark):
    dim = dim_of(spark, [("c1", "TX", "GOLD", "2026-01-01")])
    inc = spark.createDataFrame(
        [("c1", "WA", "GOLD", "2026-03-01"), ("c1", "CA", "GOLD", "2026-02-01")], SCHEMA
    )
    new = apply_scd2(dim, inc, "customer_id", TRACKED)
    assert versions(new, "c1") == [("TX", "GOLD", False), ("CA", "GOLD", False), ("WA", "GOLD", True)]
    assert new.filter("is_current").count() == 1


def test_null_to_value_counts_as_change(spark):
    dim = dim_of(spark, [("c1", None, "GOLD", "2026-01-01")])
    inc = spark.createDataFrame([("c1", "TX", "GOLD", "2026-02-01")], SCHEMA)
    assert apply_scd2(dim, inc, "customer_id", TRACKED).count() == 2


def test_delta_merge_sql_renders():
    sql = scd2_delta_merge_sql("silver.dim_customer", "updates", "customer_id", TRACKED)
    assert sql.startswith("MERGE INTO silver.dim_customer")
    assert "NOT (t.state <=> s.state)" in sql
