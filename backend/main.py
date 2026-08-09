"""Business API and replay control plane for the retail star-schema demonstration."""
import hashlib, json, os, threading, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text
from database import engine

ROOT = Path(__file__).resolve().parent
EVENT_BANK = ROOT / "original_data" / "fact_sales_normalized.csv"
UPLOADS = ROOT / "data" / "uploads" / "original"
UPLOADS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Retail Operations API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ReplayRequest(BaseModel):
    batch_size: int = Field(default=100, ge=1, le=5000)
    interval_seconds: float = Field(default=5, ge=0.1, le=3600)
class ReplayAction(BaseModel): action: Literal["pause", "resume", "stop"]

def setup():
    with engine.begin() as c:
        c.execute(text("""CREATE TABLE IF NOT EXISTS replay_state (id TINYINT PRIMARY KEY, status VARCHAR(16) NOT NULL, next_offset INT NOT NULL DEFAULT 0, batch_size INT NOT NULL DEFAULT 100, interval_seconds DECIMAL(10,2) NOT NULL DEFAULT 5, events_published INT NOT NULL DEFAULT 0, last_error TEXT NULL, updated_at DATETIME NOT NULL)"""))
        c.execute(text("""INSERT IGNORE INTO replay_state (id,status,next_offset,batch_size,interval_seconds,events_published,updated_at) VALUES (1,'idle',0,100,5,0,UTC_TIMESTAMP())"""))
        try:
            c.execute(text("ALTER TABLE replay_state ADD COLUMN events_consumed INT NOT NULL DEFAULT 0"))
        except Exception:
            pass  # Existing installations already have the migration.
        c.execute(text("CREATE TABLE IF NOT EXISTS processed_events (event_id VARCHAR(128) PRIMARY KEY, processed_at DATETIME NOT NULL)"))
setup()
def scalar(sql, params=None):
    with engine.connect() as c: return c.execute(text(sql), params or {}).scalar()
def rows(sql, params=None):
    with engine.connect() as c: return [dict(row._mapping) for row in c.execute(text(sql), params or {})]

@app.get("/health")
def health():
    try:
        scalar("SELECT 1"); return {"status":"Healthy", "database":"connected", "timestamp":datetime.now(timezone.utc)}
    except Exception as exc: return {"status":"Failed", "database":"unavailable", "detail":str(exc)}
@app.get("/dashboard")
def dashboard():
    totals=rows("SELECT COUNT(*) orders, COALESCE(SUM(total_amount),0) revenue, COUNT(DISTINCT customer_sk) customers FROM fact_sales_normalized")[0]
    trend=rows("SELECT DATE(sales_date) date, ROUND(SUM(total_amount),2) revenue FROM fact_sales_normalized GROUP BY DATE(sales_date) ORDER BY date DESC LIMIT 30")[::-1]
    products=rows("SELECT p.product_name name, ROUND(SUM(f.total_amount),2) revenue FROM fact_sales_normalized f JOIN dim_products p ON p.product_sk=f.product_sk GROUP BY p.product_name ORDER BY revenue DESC LIMIT 5")
    return {"kpis":totals,"revenue_trend":trend,"top_products":products,"freshness":scalar("SELECT MAX(sales_date) FROM fact_sales_normalized")}
@app.get("/customers")
def customers(): return {"customers":rows("SELECT customer_sk, customer_id, first_name, last_name, customer_segment FROM dim_customers LIMIT 100")}

@app.get("/analytics")
def analytics(
    task: Literal["total_revenue", "revenue_by_category", "daily_revenue", "top_customers"] = Query(...),
):
    """Return verified, data-backed analytical results for the manager UI."""
    if task == "total_revenue":
        return {"task": task, "value": float(scalar("SELECT COALESCE(SUM(total_amount),0) FROM fact_sales_normalized") or 0), "unit": "USD"}
    if task == "revenue_by_category":
        data = rows("SELECT p.category label, ROUND(SUM(f.total_amount),2) value FROM fact_sales_normalized f JOIN dim_products p ON p.product_sk=f.product_sk GROUP BY p.category ORDER BY value DESC")
    elif task == "daily_revenue":
        data = rows("SELECT DATE(sales_date) label, ROUND(SUM(total_amount),2) value FROM fact_sales_normalized GROUP BY DATE(sales_date) ORDER BY label")
    else:
        data = rows("SELECT CONCAT(c.first_name, ' ', c.last_name) label, ROUND(SUM(f.total_amount),2) value FROM fact_sales_normalized f JOIN dim_customers c ON c.customer_sk=f.customer_sk GROUP BY c.customer_sk,c.first_name,c.last_name ORDER BY value DESC LIMIT 10")
    return {"task": task, "rows": data, "empty": not data}
@app.get("/replay")
def replay_status():
    state=rows("SELECT status,next_offset,batch_size,interval_seconds,events_published,events_consumed,last_error,updated_at FROM replay_state WHERE id=1")[0]
    state["events_remaining"]=max(0,1000000-state["next_offset"]); return state
def read_batch(offset, count): return pd.read_csv(EVENT_BANK, skiprows=range(1,offset+1),nrows=count).to_dict("records")
def publish_batch():
    from controller.producer import send_event
    while True:
        state=replay_status()
        if state["status"]!="running": return
        if not state["events_remaining"]:
            with engine.begin() as c: c.execute(text("UPDATE replay_state SET status='completed',updated_at=UTC_TIMESTAMP() WHERE id=1"))
            return
        try:
            batch=read_batch(state["next_offset"],state["batch_size"])
            for sale in batch:
                sale["sales_date"]=str(sale["sales_date"]); send_event({"event_id":f"sale-{sale['sales_sk']}","event_type":"sale","payload":sale})
            with engine.begin() as c: c.execute(text("UPDATE replay_state SET next_offset=next_offset+:n,events_published=events_published+:n,updated_at=UTC_TIMESTAMP(),last_error=NULL WHERE id=1"),{"n":len(batch)})
            time.sleep(float(state["interval_seconds"]))
        except Exception as exc:
            with engine.begin() as c: c.execute(text("UPDATE replay_state SET status='failed',last_error=:e,updated_at=UTC_TIMESTAMP() WHERE id=1"),{"e":str(exc)[:2000]})
            return
@app.post("/replay/start")
def start_replay(request:ReplayRequest):
    with engine.begin() as c:
        if c.execute(text("SELECT status FROM replay_state WHERE id=1 FOR UPDATE")).scalar()=="running": raise HTTPException(409,"Replay is already running")
        c.execute(text("UPDATE replay_state SET status='running',batch_size=:b,interval_seconds=:i,last_error=NULL,updated_at=UTC_TIMESTAMP() WHERE id=1"),{"b":request.batch_size,"i":request.interval_seconds})
    threading.Thread(target=publish_batch,daemon=True).start(); return replay_status()
@app.post("/replay/control")
def control_replay(request:ReplayAction):
    status={"pause":"paused","resume":"running","stop":"stopped"}[request.action]
    with engine.begin() as c: c.execute(text("UPDATE replay_state SET status=:s,updated_at=UTC_TIMESTAMP() WHERE id=1"),{"s":status})
    if request.action=="resume": threading.Thread(target=publish_batch,daemon=True).start()
    return replay_status()
@app.get("/diagnostics")
def diagnostics():
    replay=replay_status()
    try:
        from controller.producer import kafka_available
        kafka = "Healthy" if kafka_available() else "Failed"
    except Exception:
        kafka = "Failed"
    return {"api":"Healthy","database":"Healthy","kafka":kafka,"airflow":"Configured","replay":replay,"processed_transactions":scalar("SELECT COUNT(*) FROM fact_sales_normalized")}
@app.post("/uploads/preview")
async def preview_upload(file:UploadFile=File(...)):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx",".xls"}: raise HTTPException(422,"Upload an Excel .xlsx or .xls file.")
    content=await file.read()
    if len(content)>25*1024*1024: raise HTTPException(413,"The file must be 25 MB or smaller.")
    digest=hashlib.sha256(content).hexdigest(); original=UPLOADS/f"{digest}{suffix}"
    if not original.exists(): original.write_bytes(content)
    try: frame=pd.read_excel(original)
    except Exception as exc: raise HTTPException(422,f"The Excel file could not be read: {exc}")
    return {"dataset_id":digest,"filename":file.filename,"rows":len(frame),"columns":list(frame.columns),"types":{k:str(v) for k,v in frame.dtypes.items()},"sample":json.loads(frame.head(8).to_json(orient="records",date_format="iso")),"validation":"Preview completed. Original file is preserved unchanged."}
