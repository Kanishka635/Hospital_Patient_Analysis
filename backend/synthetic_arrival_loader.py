import argparse
import json
import random
import time
from datetime import datetime, timedelta, timezone
from urllib import error, request

DEPARTMENTS = ["Emergency", "OPD", "General Ward", "Pediatrics", "ICU"]


def send_arrival(base_url: str, department: str, patient_count: int) -> dict:
    payload = {
        "period_start": datetime.now(timezone.utc).isoformat(),
        "period": "hour",
        "department": department,
        "patient_count": patient_count,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}/api/v1/arrivals",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API rejected request: {body}") from exc


def load_synthetic_data(
    interval_seconds: int,
    max_rows: int | None = None,
    base_url: str = "http://127.0.0.1:8000",
    department: str | None = None,
) -> None:
    count = 0
    while max_rows is None or count < max_rows:
        chosen_department = department or random.choice(DEPARTMENTS)
        patient_count = random.randint(5, 60)

        try:
            result = send_arrival(base_url, chosen_department, patient_count)
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"sent {patient_count} patients to {chosen_department}: {result}"
            )
            count += 1
        except Exception as exc:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {exc}")
            time.sleep(5)
            continue

        if max_rows is not None and count >= max_rows:
            break

        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send synthetic hospital arrival records into the FastAPI backend.")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between inserted rows.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional maximum rows to send before exiting.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL.")
    parser.add_argument("--department", default=None, help="Optional fixed department to use for all rows.")
    args = parser.parse_args()

    print(
        "Starting synthetic arrival stream. "
        f"Interval: {args.interval}s, base URL: {args.base_url}"
    )
    load_synthetic_data(
        interval_seconds=args.interval,
        max_rows=args.max_rows,
        base_url=args.base_url,
        department=args.department,
    )
