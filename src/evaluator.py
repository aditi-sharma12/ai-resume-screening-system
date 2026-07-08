import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.vector_store import build_vectorstore

# Load environmental configs
load_dotenv()


def get_llm():
    """
    Retrieves the active ChatGroq LLM instance.
    """
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )


def score_resume(resume_text: str, job_description: str) -> dict:
    """
    Evaluates a candidate's resume against a job description in a single one-off task.
    """
    vectorstore = build_vectorstore(resume_text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_chunks = retriever.invoke(job_description)

    if not relevant_chunks:
        return {"result": "Score: 0/100\nReasons:\n- Resume content is not relevant to the job description."}

    context = "\n".join([f"[Chunk {i+1}]: {chunk.page_content}" for i, chunk in enumerate(relevant_chunks)])

    prompt = ChatPromptTemplate.from_template(
        """You are an expert technical recruiter.
        Analyze the resume against the job description and provide:

        1. Overall Score: X/100
        2. Category Breakdown:
           - Skills Match: X/100
           - Experience Match: X/100
           - Education Match: X/100
        3. Reasons (3 bullet points)

        Job Description:
        {job_description}

        Relevant Resume Content:
        {context}

        Follow this exact format in your response.
        """
    )

    chain = prompt | get_llm()

    max_retries = 3
    backoff_factor = 4
    for attempt in range(max_retries):
        try:
            response = chain.invoke({
                "job_description": job_description,
                "context": context
            })
            return {"result": response.content}
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "503" in err_str) and attempt < max_retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                raise e


def score_candidate_from_context(filename: str, context: str, job_description: str) -> dict:
    """
    Evaluates a candidate's match using pre-retrieved context chunks from the database.
    """
    prompt = ChatPromptTemplate.from_template(
        """You are an expert technical recruiter.
        Analyze the resume content for candidate '{filename}' against the job description and provide:

        1. Overall Score: X/100
        2. Category Breakdown:
           - Skills Match: X/100
           - Experience Match: X/100
           - Education Match: X/100
        3. Reasons (3 bullet points explaining why this score was given based on the resume)

        Job Description:
        {job_description}

        Relevant Resume Content for {filename}:
        {context}

        Follow this exact format in your response.
        """
    )

    chain = prompt | get_llm()

    max_retries = 3
    backoff_factor = 4
    for attempt in range(max_retries):
        try:
            response = chain.invoke({
                "filename": filename,
                "job_description": job_description,
                "context": context
            })
            return {"result": response.content}
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "503" in err_str) and attempt < max_retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                raise e
