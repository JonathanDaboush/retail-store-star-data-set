from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller.producer import send_event
from controller.test_controller import get_customers
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/customers")
def request_customers():
    
    return {"customers": get_customers()}