
from datetime import datetime
from pathlib import Path
import sqlite3

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------
# App setup
# --------------------------------------------------

app = FastAPI(
    title="Hospital Patient Arrival API",
    description="API for storing and analyzing hospital patient arrival counts.",
    version="1.0.0",
)

# Keep the database inside the backend folder
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "incoming_counts.db"


# --------------------------------------------------
# Database setup
# --------------------------------------------------

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS arrivals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TEXT NOT NULL,
                period TEXT NOT NULL CHECK (period IN ('hour', 'day')),
                department TEXT NOT NULL,
                patient_count INTEGER NOT NULL CHECK (patient_count >= 0),
                UNIQUE (period_start, period, department)
            )
            """
        )


initialize_database()


# --------------------------------------------------
# Request data validation
# --------------------------------------------------

class ArrivalCount(BaseModel):
    period_start: datetime
    period: str
    department: str = Field(min_length=1, max_length=100)
    patient_count: int = Field(ge=0)

    @field_validator("period")
    @classmethod
    def validate_period(cls, value):
        if value not in ("hour", "day"):
            raise ValueError("period must be 'hour' or 'day'")
        return value

    @field_validator("period_start")
    @classmethod
    def validate_timezone(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "period_start must include a timezone, such as +05:30"
            )
        return value

    @field_validator("department")
    @classmethod
    def clean_department(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("department cannot be blank")
        return value


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "hospital-arrival-api",
    }


# --------------------------------------------------
# Receive and store arrival counts
# --------------------------------------------------

@app.post("/api/v1/arrivals")
def add_arrival_count(record: ArrivalCount):
    timestamp = record.period_start.isoformat()

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO arrivals
                    (period_start, period, department, patient_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(period_start, period, department)
                DO UPDATE SET patient_count = excluded.patient_count
                """,
                (
                    timestamp,
                    record.period,
                    record.department,
                    record.patient_count,
                ),
            )
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail="Could not save arrival data.",
        ) from error

    return {
        "message": "Arrival count saved successfully",
        "record": {
            "period_start": timestamp,
            "period": record.period,
            "department": record.department,
            "patient_count": record.patient_count,
        },
    }


# --------------------------------------------------
# Retrieve arrival records
# --------------------------------------------------

@app.get("/api/v1/arrivals")
def get_arrivals(
    period: str | None = Query(default=None, pattern="^(hour|day)$"),
    department: str | None = Query(default=None, min_length=1),
    limit: int = Query(default=100, ge=1, le=5000),
):
    sql = "SELECT * FROM arrivals WHERE 1=1"
    parameters = []

    if period is not None:
        sql += " AND period = ?"
        parameters.append(period)

    if department is not None:
        sql += " AND department = ?"
        parameters.append(department.strip())

    sql += " ORDER BY period_start DESC LIMIT ?"
    parameters.append(limit)

    with get_connection() as connection:
        rows = connection.execute(sql, parameters).fetchall()

    return {
        "count": len(rows),
        "records": [dict(row) for row in rows],
    }


# --------------------------------------------------
# Summary statistics
# --------------------------------------------------

@app.get("/api/v1/summary")
def get_summary(
    period: str = Query(default="day", pattern="^(hour|day)$"),
    department: str | None = Query(default=None, min_length=1),
):
    sql = """
        SELECT
            COUNT(*) AS number_of_periods,
            COALESCE(SUM(patient_count), 0) AS total_patients,
            COALESCE(AVG(patient_count), 0) AS average_patients,
            COALESCE(MIN(patient_count), 0) AS minimum_patients,
            COALESCE(MAX(patient_count), 0) AS maximum_patients
        FROM arrivals
        WHERE period = ?
    """
    parameters = [period]

    if department is not None:
        sql += " AND department = ?"
        parameters.append(department.strip())

    with get_connection() as connection:
        row = connection.execute(sql, parameters).fetchone()

    return {
        "period": period,
        "department": department,
        "number_of_periods": row["number_of_periods"],
        "total_patients": row["total_patients"],
        "average_patients": round(row["average_patients"], 2),
        "minimum_patients": row["minimum_patients"],
        "maximum_patients": row["maximum_patients"],
    }
