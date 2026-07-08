# AI Resume Screening System

A clean, high-speed, AI-powered Applicant Tracking System (ATS) that helps recruiters organize candidate resumes, paste job descriptions, and instantly find the best-matched profiles.

---

## 🌟 Key Features

* **📁 Resume Library**: Upload and manage candidate resumes (PDF format, up to 30 at a time) in a secure, local directory.
* **🔍 Find Best Matches**: Paste a job description and instantly rank all candidates in your directory based on Skills, Experience, and Education.
* **🎯 Quick Check**: Drag-and-drop a single resume to evaluate it on-the-fly without saving it to your directory.
* **⚡ Interactive Scorecards**: View detailed bullet points highlighting the candidate's strengths, gaps, and compatibility scores out of 100.
* **⚙️ Simple Settings**: Paste your secure AI Access Key directly inside the sidebar to start screening immediately.

---

## 🛠️ Built With

* **Frontend Dashboard**: Streamlit (Python)
* **AI Evaluation Engine**: Groq Cloud & Llama 3.3 (Blazing fast processing speed)
* **Local Database**: FAISS & SentenceTransformers (Stores candidate profiles locally on your computer—completely private)

---

## 🚀 Getting Started (Setup)

Follow these simple steps to run the application on your computer:

### 1. Set Up Your Environment
Open your terminal and run the following commands to set up the project:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install the required software packages
pip install -r requirements.txt
```

### 2. Configure Your Access Key
1. Create a file named `.env` in the root of the project directory.
2. Enter your Groq API key:
   ```env
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

### 3. Launch the Application
Start the local server by running:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser!