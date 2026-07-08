import streamlit as st
import os, time, re, tempfile, json, shutil, datetime
import pandas as pd
import altair as alt
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="AI Resume Screening System", page_icon="🧑‍💼", layout="wide")

for key, default in [("screen_results", None), ("screen_run", False)]:
    st.session_state.setdefault(key, default)

CANDIDATES_JSON = "candidates.json"

# styling 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer {visibility:hidden;}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-top: -3.5rem !important;
    margin-bottom: 1.2rem;
    color: #fff;
}
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; line-height: 1.2; }
.hero p  { margin: .3rem 0 0; opacity: .9; font-size: 1rem; }

/* Cards */
.card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 1rem 1.2rem; }

/* Score box */
.score-box {
    display: flex; align-items: center; gap: 1rem;
    background: rgba(79,70,229,0.1); border: 1px solid rgba(79,70,229,0.2);
    border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    flex-wrap: wrap;
}
.score-ring {
    width: 72px; height: 72px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 800; font-size: 1.3rem; flex-shrink: 0;
}
.pill   { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 700; }
.reason { background: rgba(255,255,255,0.02); border-left: 3px solid #4F46E5; border-radius: 8px; padding: .5rem .8rem; margin-bottom: .4rem; }
.empty  { text-align: center; padding: 2.5rem 1rem; border: 1.5px dashed rgba(255,255,255,0.12); border-radius: 14px; color: #64748B; background: rgba(255,255,255,0.01); }

/* Library rows */
.lib-row {
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: .55rem .9rem; margin-bottom: .5rem;
}
.lib-name { font-weight: 600; font-size: .92rem; }
.lib-date { font-size: .78rem; color: #94A3B8; margin-top: 2px; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#131237,#0F172A) !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,[data-testid="stSidebar"] caption { color: #F8FAFC !important; }
[data-testid="stSidebar"] p,[data-testid="stSidebar"] small { color: #CBD5E1 !important; }

/* Responsive breakpoints */
@media (max-width: 768px) {
    .hero { padding: 1.2rem 1.2rem; margin-top: -2rem !important; border-radius: 12px; }
    .hero h1 { font-size: 1.4rem; }
    .hero p  { font-size: 0.85rem; }
    .score-box { flex-direction: column; align-items: flex-start; padding: 1rem; }
    .score-ring { width: 58px; height: 58px; font-size: 1.1rem; }
}

@media (max-width: 480px) {
    .hero { padding: 1rem; margin-top: -1.5rem !important; border-radius: 10px; }
    .hero h1 { font-size: 1.15rem; }
    .hero p  { font-size: 0.8rem; }
    .score-ring { width: 50px; height: 50px; font-size: 1rem; }
    .reason { font-size: 0.82rem; }
}
</style>
""", unsafe_allow_html=True)


# data helpers
def load_candidates():
    if os.path.exists(CANDIDATES_JSON):
        try:
            return json.load(open(CANDIDATES_JSON))
        except Exception:
            return []
    return []

def save_candidates(items):
    json.dump(items, open(CANDIDATES_JSON, "w"), indent=4)

def add_candidate_meta(filename):
    items = load_candidates()
    if not any(c["filename"].lower() == filename.lower() for c in items):
        items.append({"filename": filename, "indexed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        save_candidates(items)

def remove_candidate_meta(filename):
    """Remove a single candidate's metadata entry from candidates.json."""
    items = load_candidates()
    items = [c for c in items if c["filename"].lower() != filename.lower()]
    save_candidates(items)

def delete_candidate(filename):
    """Remove a single candidate from the metadata store and, if possible, the vector store."""
    remove_candidate_meta(filename)
    try:
        # Best-effort: only works if src.vector_store exposes a per-file delete helper.
        from src.vector_store import delete_resume_from_vectorstore
        delete_resume_from_vectorstore(filename)
    except ImportError:
        pass
    except Exception as e:
        st.warning(f"Removed '{filename}' from the library, but couldn't clean up its indexed data: {e}")

def reset_database():
    if os.path.exists(CANDIDATES_JSON):
        os.remove(CANDIDATES_JSON)
    shutil.rmtree("./vectorstore", ignore_errors=True)

def has_api_key():
    return bool(os.getenv("GROQ_API_KEY"))


# score parsing 
def parse_score_response(text):
    def find(pattern, default=0):
        m = re.search(pattern, text, re.IGNORECASE)
        return int(m.group(1)) if m else default

    overall = find(r"Overall Score:\s*(\d+)") or find(r"Score:\s*(\d+)")
    skills = find(r"Skills(?: Match)?:\s*(\d+)")
    experience = find(r"Experience(?: Match)?:\s*(\d+)")
    education = find(r"Education(?: Match)?:\s*(\d+)")

    reasons = []
    sec = re.search(r"Reasons:?(.*)", text, re.DOTALL | re.IGNORECASE)
    if sec:
        bullets = re.findall(r"(?:^|\n)[-*\d\.]+\s*(.+)", sec.group(1))
        if bullets:
            reasons = [b.strip() for b in bullets if b.strip()]
        else:
            reasons = [l.strip().lstrip("-*• ") for l in sec.group(1).split("\n") if len(l.strip()) > 10][:4]

    return {"overall": overall, "skills": skills, "experience": experience, "education": education, "reasons": reasons}


# UI helpers 
def score_color(v):
    return "#059669" if v >= 75 else "#D97706" if v >= 50 else "#DC2626"

def score_label(v):
    return "Strong Match" if v >= 75 else "Partial Match" if v >= 50 else "Weak Match"

def render_scorecard(p):
    color = score_color(p["overall"])
    st.markdown(f"""
        <div class="score-box">
            <div class="score-ring" style="background:{color};">{p['overall']}</div>
            <div>
                <div style="font-weight:700; font-size:1.1rem;">Overall Match: {p['overall']}/100</div>
                <span class="pill" style="background:{color}22; color:{color};">● {score_label(p['overall'])}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, label, icon, key in zip([c1, c2, c3], ["Skills", "Experience", "Education"], ["💻", "💼", "🎓"],
                                      ["skills", "experience", "education"]):
        with col:
            st.metric(f"{icon} {label} Match", f"{p[key]}/100")

    st.markdown("**🔍 Strengths & Gaps**")
    if p["reasons"]:
        for r in p["reasons"]:
            st.markdown(f"<div class='reason'>✦ {r}</div>", unsafe_allow_html=True)
    else:
        st.info("No details returned for this candidate.")

def empty_state(icon, title, sub):
    st.markdown(f"<div class='empty'><div style='font-size:2rem;'>{icon}</div>"
                f"<b>{title}</b><div style='font-size:.9rem;'>{sub}</div></div>", unsafe_allow_html=True)


# sidebar 
with st.sidebar:
    st.title("🧑‍💼 Screening System")
    st.caption("AI-powered resume screening")
    st.write("")
    page = st.radio("Menu", ["📁 Resume Library", "🔍 Find Best Matches", "🎯 Quick Check"], label_visibility="collapsed", key="sidebar_nav_menu")
    st.markdown("---")

    if not has_api_key():
        st.warning("⚠️ Setup needed to get started.")
        key = st.text_input("Access Key", type="password", help="Ask your admin for this key if you don't have one.")
        if key:
            os.environ["GROQ_API_KEY"] = key
            st.success("✅ You're all set for this session!")
    else:
        st.success("✅ Ready to go!")

    st.markdown(f"""
        <div style="text-align: center; margin: 1.5rem 0; background: rgba(255,255,255,0.05); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em;">Resumes in Library</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: #818CF8; margin-top: 0.3rem; line-height: 1;">{len(load_candidates())}</div>
        </div>
    """, unsafe_allow_html=True)
    st.info("🤖 Keeps your resumes organized and matches them against job openings using AI.")


# header 
st.markdown("<div class='hero'><h1>💼 AI Resume Screening System</h1>"
            "<p>Organize resumes, match them to your job openings, and find your best candidates in seconds.</p></div>",
            unsafe_allow_html=True)

# page content
if page == "📁 Resume Library":
    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        st.subheader("Add Resumes (up to 30 at a time)")
        with st.form("index_form", clear_on_submit=True):
            uploaded_pdfs = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)
            submitted = st.form_submit_button("Add to Library", type="primary", use_container_width=True)

        if submitted:
            if not has_api_key():
                st.error("🔑 Please add your access key in the settings panel first.")
            elif not uploaded_pdfs:
                st.warning("Please upload at least one resume PDF.")
            else:
                from src.parser import extract_text_from_pdf
                from src.vector_store import add_resume_to_vectorstore

                progress, status = st.progress(0), st.empty()
                success = 0
                for i, pdf in enumerate(uploaded_pdfs):
                    status.text(f"Processing {i+1}/{len(uploaded_pdfs)}: {pdf.name}...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf.getbuffer())
                        path = tmp.name
                    try:
                        text = extract_text_from_pdf(path)
                        add_resume_to_vectorstore(text, pdf.name)
                        add_candidate_meta(pdf.name)
                        success += 1
                    except Exception as e:
                        st.error(f"Couldn't add {pdf.name}: {e}")
                    finally:
                        os.path.exists(path) and os.remove(path)
                    progress.progress((i + 1) / len(uploaded_pdfs))

                status.success(f"🎉 Added {success} of {len(uploaded_pdfs)} resumes to your library!")
                st.session_state.screen_results = None
                st.session_state.screen_run = False
                st.rerun()

    with col2:
        st.subheader("Resumes in Your Library")
        with st.container(border=True):
            candidates = sorted(load_candidates(), key=lambda c: c["filename"].lower())
            if candidates:
                # Scrollable list with a per-resume delete button.
                with st.container(height=340):
                    for c in candidates:
                        row_l, row_r = st.columns([5, 1], vertical_alignment="center")
                        with row_l:
                            st.markdown(
                                f"<div class='lib-row'><div>"
                                f"<div class='lib-name'>📄 {c['filename']}</div>"
                                f"<div class='lib-date'>Added {c['indexed_at']}</div>"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )
                        with row_r:
                            if st.button("🗑️", key=f"delete_{c['filename']}", help=f"Remove {c['filename']}", use_container_width=True):
                                delete_candidate(c["filename"])
                                st.session_state.screen_results = None
                                st.session_state.screen_run = False
                                st.toast(f"Removed '{c['filename']}' from the library.")
                                st.rerun()

                st.markdown("")
                if st.button("🚨 Clear Library", use_container_width=True):
                    reset_database()
                    st.session_state.screen_results = None
                    st.session_state.screen_run = False
                    st.success("Your resume library has been cleared.")
                    st.rerun()
            else:
                empty_state("📭", "Your library is empty", "Upload resumes to get started!")

# TAB 2: Screen 
elif page == "🔍 Find Best Matches":
    candidates = load_candidates()
    if not candidates:
        empty_state("🗂️", "No resumes yet", "Add resumes in the Resume Library tab first.")
    else:
        st.subheader("Job Requirements")
        with st.container(border=True):
            jd = st.text_area("Paste your job description", height=200,
                               placeholder="Enter job role, required skills, and qualifications...",
                               key="screen_jd_input_vertical")
            if st.button("🔍 Find & Rank Best Candidates", type="primary", use_container_width=True, key="persistent_screen_search_btn"):
                st.session_state.screen_run = True
                st.session_state.screen_results = None

        st.write("")
        st.markdown("---")

        if st.session_state.screen_run:
            st.subheader("Top Candidates")
            with st.container(border=True):
                if st.session_state.screen_results is None:
                    if not has_api_key():
                        st.error("🔑 Please add your access key in the settings panel first.")
                        st.session_state.screen_run = False
                    elif not jd:
                        st.warning("Please provide a job description to match candidates against.")
                        st.session_state.screen_run = False
                    else:
                        with st.spinner("Reviewing resumes and scoring each candidate..."):
                            from src.vector_store import retrieve_candidates_from_db
                            from src.evaluator import score_candidate_from_context

                            k = min(30, max(10, len(candidates) * 5))
                            context = retrieve_candidates_from_db(jd, k=k)

                            if not context:
                                st.warning("No matching candidates found.")
                                st.session_state.screen_run = False
                            else:
                                results, progress, status = [], st.progress(0), st.empty()
                                for i, (filename, ctx) in enumerate(context.items()):
                                    status.text(f"Reviewing {i+1}/{len(context)}: {filename}...")
                                    try:
                                        raw = score_candidate_from_context(filename, ctx, jd)["result"]
                                        p = parse_score_response(raw)
                                    except Exception as e:
                                        raw, p = f"Evaluation error: {e}", {
                                            "overall": 0, "skills": 0, "experience": 0, "education": 0,
                                            "reasons": [f"Failed to score: {e}"]
                                        }
                                    results.append({
                                        "Filename": filename, "Score": p["overall"],
                                        "Skills Match": p["skills"], "Experience Match": p["experience"],
                                        "Education Match": p["education"], "Details": raw, "Parsed": p
                                    })
                                    progress.progress((i + 1) / len(context))
                                    time.sleep(2)

                                status.success("🎉 All candidates reviewed!")
                                df = pd.DataFrame(results)[
                                    ["Filename", "Score", "Skills Match", "Experience Match", "Education Match"]
                                ].sort_values("Score", ascending=False).reset_index(drop=True)
                                st.session_state.screen_results = {"results": results, "df": df}

                if st.session_state.screen_results:
                    results, df = st.session_state.screen_results["results"], st.session_state.screen_results["df"]

                    m1, m2, m3 = st.columns(3)
                    m1.metric("👥 Reviewed", len(df))
                    m2.metric("🏆 Top Score", int(df["Score"].max()) if len(df) else 0)
                    m3.metric("📊 Average", int(df["Score"].mean()) if len(df) else 0)

                    st.dataframe(df, use_container_width=True, hide_index=True, column_config={
                        c: st.column_config.ProgressColumn(c, format="%d", min_value=0, max_value=100)
                        for c in ["Score", "Skills Match", "Experience Match", "Education Match"]
                    })

                    st.markdown("### 🔍 View a Candidate's Full Scorecard")
                    pick = st.selectbox("Choose a candidate:", df["Filename"].tolist(), label_visibility="collapsed", key="persistent_screen_inspector_selectbox")
                    cand = next(r for r in results if r["Filename"] == pick)
                    p = cand["Parsed"]

                    if any(r.startswith("Failed to score:") for r in p["reasons"]):
                        st.warning("⚠️ Scorecard unavailable — the system was busy. Try running the search again.")
                        with st.expander("🛠️ View error details"):
                            for r in p["reasons"]:
                                st.write(r)
                    else:
                        render_scorecard(p)
        else:
            empty_state("🧭", "Ready when you are", "Enter a job description and click search to view matches.")

# TAB 3: Quick Check 
else:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("📋 Job & Resume")
        with st.container(border=True):
            jd_input = st.text_area("Paste Job Description", height=230,
                                     placeholder="Enter job role, required skills, and qualifications...",
                                     key="onthefly_jd")
            resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"],
                                            help="Checked instantly, not saved to your library.",
                                            key="onthefly_resume")
            check = st.button("⚡ Check Candidate", type="primary", use_container_width=True, key="onthefly_score_search_btn")

    with col2:
        st.subheader("⚡ Scorecard")
        with st.container(border=True):
            if check:
                if not has_api_key():
                    st.error("🔑 Please add your access key in the settings panel first.")
                elif not jd_input:
                    st.warning("⚠️ Please provide a job description.")
                elif not resume_file:
                    st.warning("⚠️ Please upload a candidate resume PDF.")
                else:
                    with st.spinner("Reviewing resume..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(resume_file.getbuffer())
                            path = tmp.name
                        try:
                            from src.parser import extract_text_from_pdf
                            from src.evaluator import score_resume
                            text = extract_text_from_pdf(path)
                            result = score_resume(text, jd_input)
                        except Exception as e:
                            st.error(f"We couldn't review this resume: {e}")
                            result = None
                        finally:
                            os.path.exists(path) and os.remove(path)

                        if result and "result" in result:
                            raw = result["result"]
                            p = parse_score_response(raw)
                            st.success("✅ Review Complete!")
                            render_scorecard(p)

                            chart_df = pd.DataFrame({
                                "Category": ["Skills", "Experience", "Education"],
                                "Score": [p["skills"], p["experience"], p["education"]]
                            })
                            chart = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                                x=alt.X("Category:N", axis=alt.Axis(labelAngle=0)),
                                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100])),
                                color=alt.Color("Category:N", legend=None, scale=alt.Scale(scheme="blues"))
                            ).properties(height=200)
                            st.altair_chart(chart, use_container_width=True)
            else:
                empty_state("⚡", "Awaiting input", "Upload a resume and paste the job description to check a candidate.")