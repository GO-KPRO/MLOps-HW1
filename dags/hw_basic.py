from datetime import datetime, timedelta
from airflow.sdk import dag, task, Param, BaseHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from zoneinfo import ZoneInfo
import batch_common as bc
import requests


@dag(
    dag_id="hw_basic_batch_inference",
    start_date=datetime(2026, 9, 1, tzinfo=ZoneInfo("Europe/Moscow")),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    params={
        "business_date": Param("2026-09-01", type="string", format="date"),
        "batch_size": Param(32, type="integer", minimum=1, maximum=1000),
    },
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=5),
        "execution_timeout": timedelta(minutes=2),
    },
)
def hw_basic_batch_inference():

    @task
    def extract(**context):
        explicit = context["params"]["business_date"]
        interval_start = context.get("data_interval_start")
        day = bc.business_date(explicit, interval_start)

        conn = BaseHook.get_connection("source_api")
        host = conn.host

        rows_resp = requests.get(f"{host}/basic/{day}.json", timeout=10)
        rows_resp.raise_for_status()
        rows_payload = rows_resp.json()
        if rows_payload["business_date"] != day:
            raise ValueError(f"business_date mismatch: requested {day}")
        rows = rows_payload["rows"]

        model_resp = requests.get(f"{host}/models/v1.json", timeout=10)
        model_resp.raise_for_status()
        model = model_resp.json()
        model_sha256 = bc.digest(model_resp.content)

        return bc.save({
            "business_date": day,
            "rows": rows,
            "model": model,
            "model_sha256": model_sha256,
        })

    @task(retries=0)
    def infer(extract_ref, **context):
        payload = bc.read(extract_ref)
        day = payload["business_date"]
        rows = bc.validate_rows(payload["rows"], day)
        batch_size = context["params"]["batch_size"]
        predictions = bc.infer_batches(rows, payload["model"], batch_size)
        expected_ids = [row["customer_id"] for row in rows]
        bc.validate_predictions(predictions, expected_ids)

        return bc.save({
            "business_date": day,
            "model_version": "v1",
            "predictions": predictions,
            "model_sha256": payload["model_sha256"],
        })

    @task
    def publish(infer_ref):
        payload = bc.read(infer_ref)
        conn = PostgresHook(postgres_conn_id="warehouse").get_conn()
        try:
            bc.replace_day(
                conn,
                payload["business_date"],
                payload["model_version"],
                payload["predictions"],
            )
        finally:
            conn.close()

        return {
            "business_date": payload["business_date"],
            "model_version": payload["model_version"],
            "rows": len(payload["predictions"]),
        }

    extracted = extract()
    inferred = infer(extracted)
    publish(inferred)


hw_basic_batch_inference()