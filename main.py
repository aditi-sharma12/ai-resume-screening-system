from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os

from src.parser import extract_text_from_pdf
from src.evaluator import score_resume

app = FastAPI(title="RAG Candidate Scoring Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "RAG Candidate Scoring Engine is running"}


@app.post("/score")
async def score(resume: UploadFile, job_description: str = Form(...)):
    temp_path = f"temp_{resume.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(resume.file, f)
    resume_text = extract_text_from_pdf(temp_path)
    result = score_resume(resume_text, job_description)
    os.remove(temp_path)
    return result


@app.post("/rank")
async def rank_resumes(resumes: List[UploadFile], job_description: str = Form(...)):
    results = []
    for resume in resumes:
        temp_path = f"temp_{resume.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(resume.file, f)
        resume_text = extract_text_from_pdf(temp_path)
        result = score_resume(resume_text, job_description)
        results.append({"filename": resume.filename, "result": result["result"]})
        os.remove(temp_path)
    return {"rankings": sorted(results, key=lambda x: x["result"], reverse=True)}