import csv
import io
import os
import shutil
import tempfile
import uuid

import polars as pl
import requests
import uvicorn
from datasets import Dataset
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

script_dir = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(script_dir, "static")
templates_path = os.path.join(script_dir, "templates")

app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

SERVER_URL = "http://localhost:8000/process"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("main_interface.html", {"request": request})


@app.post("/api/run")
async def run_experiment(file: UploadFile = File(...), question: str = Form(...)):
    temp_dir_name = f"temp_dataset_{uuid.uuid4()}"
    temp_path = os.path.join(tempfile.gettempdir(), temp_dir_name)

    try:
        contents = await file.read()

        sample_text = contents[:2048].decode("utf-8", errors="ignore")
        sniffer = csv.Sniffer()
        seperator = sniffer.sniff(sample_text, delimiters=[",", ";", "\t", "|"]).delimiter

        file_obj = io.BytesIO(contents)

        if file.filename.endswith(".csv"):
            df = pl.read_csv(file_obj, separator=seperator, truncate_ragged_lines=True, ignore_errors=True)
        else:
            return JSONResponse(content={"error": "Invalid file format. Use a CSV file format"}, status_code=400)

        headers = df.columns
        rows = [[str(item) if item is not None else "" for item in row] for row in df.rows()]

        data_structure = [
            {"id": str(0), "table": {"header": headers, "rows": rows, "name": file.filename}, "question": question, "answers": None}
        ]

        hf_dataset = Dataset.from_list(data_structure)

        hf_dataset.save_to_disk(temp_path)

        payload = {"dataset_path": temp_path, "question": question}

        header = {"x-secret-token": "Leap_tool_secret_token"}

        response = requests.post(SERVER_URL, json=payload, headers=header)

        if response.status_code != 200:
            return JSONResponse(content={"error": f"Backend Error: {response.text}"}, status_code=500)

        return response.json()

    except Exception as e:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("interface.main_interface:app", host="0.0.0.0", port=8082, reload=True)
