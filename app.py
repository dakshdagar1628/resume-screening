"""
app.py  -  MODULE 3: Streamlit web interface
Run from a terminal inside this folder:   streamlit run app.py
"""
import os
import glob
import html
import tempfile
import pandas as pd
import streamlit as st
from extractor import parse_resume, extract_text
from matcher import rank_candidates, sort_ranked, SORT_OPTIONS

NOTE = ("Human review required: this score only measures keyword overlap with the job description. "
        "It is NOT a measure of a candidate's ability or suitability. "
        "The final decision always stays with a human recruiter.")

st.set_page_config(page_title="AI Resume Screening", layout="wide")
st.title("AI Resume Screening and Candidate Ranking")
st.warning(NOTE)                       # clear human-review note, always visible


def chips(items, color):
    """Small coloured tags for matched / missing skills."""
    if not items:
        return "<i>none</i>"
    return " ".join(f"<span style='background:{color};color:#000;padding:2px 8px;border-radius:10px;"
                    f"margin:2px;display:inline-block'>{html.escape(s)}</span>" for s in items)


def save_upload(f, folder):
    path = os.path.join(folder, os.path.basename(f.name))
    with open(path, "wb") as out:
        out.write(f.getvalue())
    return path


# ---------------- sidebar: inputs ----------------
with st.sidebar:
    st.header("1. Resumes")
    uploads = st.file_uploader("Upload resumes (PDF or DOCX)", accept_multiple_files=True)
    use_sample = st.checkbox("Also use the sample resumes")

    st.header("2. Job description")
    sample_jds = sorted(glob.glob("sample_data/job_descriptions/*.txt"))
    sample_choice = st.selectbox("Sample JD", ["(none)"] + [os.path.basename(p) for p in sample_jds])
    jd_file = st.file_uploader("Upload JD (txt, pdf, docx)", type=["txt", "pdf", "docx"])
    jd_pasted = st.text_area("...or paste the JD here", height=180)

    st.header("3. Sort")
    sort_option = st.selectbox("Sort candidates by", list(SORT_OPTIONS.keys()))

# ---------------- read the JD (pasted > uploaded > sample) ----------------
tmp = tempfile.mkdtemp()
jd_text = jd_pasted.strip()
if not jd_text and jd_file is not None:
    p = save_upload(jd_file, tmp)
    try:
        jd_text = open(p, encoding="utf-8", errors="ignore").read() if p.lower().endswith(".txt") else extract_text(p)
    except Exception as e:
        st.error(f"Job description error: {e}")
if not jd_text and sample_choice != "(none)":
    jd_text = open(os.path.join("sample_data/job_descriptions", sample_choice)).read()

# ---------------- collect resume files ----------------
paths = [save_upload(f, tmp) for f in uploads]
if use_sample:
    paths += sorted(glob.glob("sample_data/resumes/*"))

if not paths or not jd_text.strip():
    st.info("Add resumes (upload or tick the sample box) and a job description in the sidebar to start.")
    st.stop()

# ---------------- parse + rank ----------------
parsed = [parse_resume(p) for p in paths]
bad = [(p["file"], p["error"]) for p in parsed if p["error"]]
for fname, err in bad:
    st.error(f"Skipped {fname}: {err}")          # clear message for unreadable / unsupported files
if len(bad) == len(parsed):
    st.stop()

kw, ranked = rank_candidates(parsed, jd_text)
if not (kw["required"] or kw["optional"] or kw["extra"]):
    st.warning("No known skills were found in this job description, so all scores will be 0.")

st.subheader("Keywords found in the job description")
c1, c2, c3 = st.columns(3)
c1.markdown("**Required**"); c1.markdown(chips(kw["required"], "#cfe8ff"), unsafe_allow_html=True)
c2.markdown("**Optional**"); c2.markdown(chips(kw["optional"], "#e6e6e6"), unsafe_allow_html=True)
c3.markdown("**Extra (TF-IDF)**"); c3.markdown(chips(kw["extra"], "#f3e5ff"), unsafe_allow_html=True)

# ---------------- ranked table (sortable) ----------------
st.subheader("Ranked candidates")
view = sort_ranked(ranked, sort_option).copy()
table = view[["name", "final_score", "skill_score", "similarity", "years"]].copy()
table["matched"] = view["matched"].apply(len)
table["missing"] = view["missing"].apply(len)
table.insert(0, "rank", range(1, len(table) + 1))
st.dataframe(table, hide_index=True)

# ---------------- candidate detail view ----------------
st.subheader("Candidate detail")
labels = {f"{r['name']}  (score {r['final_score']})": r["file"] for _, r in view.iterrows()}
choice = st.selectbox("Pick a candidate", list(labels.keys()))
row = ranked[ranked["file"] == labels[choice]].iloc[0]
cand = next(p for p in parsed if p["file"] == row["file"])

st.markdown(f"### {row['name']} - final score {row['final_score']}")
st.write("**Why this score:** " + row["reason"])
st.markdown("**Matched skills:** " + chips(row["matched"], "#b7e4c7"), unsafe_allow_html=True)
st.markdown("**Missing skills:** " + chips(row["missing"], "#ffc9c9"), unsafe_allow_html=True)
st.write(f"**Years of experience (detected):** {cand['years_experience']}")
st.write("**Education:** " + cand["education"])
st.write("**Experience:** " + cand["experience"])
st.write("**Certifications:** " + cand["certifications"])
st.write("**Projects:** " + cand["projects"])
st.warning(NOTE)
