"""
extractor.py  -  MODULE 1: resume extraction
Reads PDF/DOCX resumes, cleans the text, and pulls out structured fields.
"""
import os
import re
import pdfplumber
import docx

SUPPORTED = (".pdf", ".docx")   # file types we can read


# ---------------------------------------------------------------
# Step 1: read raw text from a file (with clear error messages)
# ---------------------------------------------------------------
def extract_text(path):
    """Read raw text from a PDF or DOCX. Raises clear errors for bad files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{ext}'. Only PDF and DOCX are allowed.")
    try:
        if ext == ".pdf":
            # pdfplumber reads text page by page
            with pdfplumber.open(path) as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        else:
            # python-docx reads paragraphs; resumes often hide text in tables too
            d = docx.Document(path)
            parts = [p.text for p in d.paragraphs]
            for t in d.tables:
                for row in t.rows:
                    for cell in row.cells:
                        parts.append(cell.text)
            text = "\n".join(parts)
    except Exception as e:
        raise ValueError(f"Could not read file (corrupted or unreadable): {e}")
    if not text.strip():
        raise ValueError("No text found (file is empty or a scanned image).")
    return text


# ---------------------------------------------------------------
# Step 2: clean and normalize the text
# ---------------------------------------------------------------
def clean_text(text):
    """Normalize text: fix odd characters, extra spaces, keep line breaks."""
    text = text.replace("\u2022", " ").replace("\u2013", "-").replace("\u2014", "-")  # bullets, dashes
    text = re.sub(r"[ \t]+", " ", text)        # collapse spaces/tabs into one space
    text = re.sub(r"\n\s*\n+", "\n", text)     # remove blank lines
    return text.strip()


# ---------------------------------------------------------------
# Step 3: skills dictionary (canonical skill -> list of aliases)
# ---------------------------------------------------------------
SKILLS = {
    # programming
    "python": [], "java": [], "javascript": ["js"], "typescript": [], "c++": [], "c#": [], "php": [], "go": ["golang"],
    # data / ML
    "sql": [], "pandas": [], "numpy": [], "machine learning": ["ml"], "deep learning": ["dl"],
    "scikit-learn": ["sklearn", "scikit learn"], "tensorflow": [], "keras": [], "pytorch": [],
    "nlp": ["natural language processing"], "computer vision": [], "statistics": ["statistical analysis"],
    "data visualization": ["data visualisation"], "data analysis": ["data analytics"],
    "power bi": ["powerbi"], "tableau": [], "excel": ["ms excel"], "matplotlib": [], "seaborn": [],
    "jupyter notebook": ["jupyter"], "big data": [], "spark": ["pyspark"], "hadoop": [],
    # web
    "html": ["html5"], "css": ["css3"], "react": ["reactjs", "react.js"], "angular": [], "vue": ["vue.js"],
    "node.js": ["nodejs", "node js"], "express": ["express.js", "expressjs"], "django": [], "flask": [],
    "rest apis": ["rest api", "restful api", "restful apis", "rest"], "graphql": [], "tailwind": [], "bootstrap": [],
    # databases
    "mongodb": ["mongo"], "mysql": [], "postgresql": ["postgres"], "firebase": [], "redis": [],
    # devops / cloud
    "git": [], "github": [], "docker": [], "kubernetes": ["k8s"], "aws": ["amazon web services"],
    "azure": [], "gcp": ["google cloud"], "ci/cd": ["cicd", "ci cd"], "linux": [], "agile": ["scrum"],
    # soft skills
    "communication": [], "problem solving": ["problem-solving"], "teamwork": ["team work", "collaboration"],
    "leadership": [], "time management": [],
}

# reverse lookup: any alias -> canonical name (so "ML" and "machine learning" are the same skill)
ALIAS_TO_CANON = {}
for _canon, _aliases in SKILLS.items():
    ALIAS_TO_CANON[_canon] = _canon
    for _a in _aliases:
        ALIAS_TO_CANON[_a] = _canon


def extract_skills(text):
    """Dictionary matching with word boundaries. Returns a set of canonical skills."""
    t = text.lower()
    found = set()
    for term, canon in ALIAS_TO_CANON.items():
        # the lookbehind/lookahead stops 'java' from matching inside 'javascript'
        pattern = r"(?<![a-z0-9+#.])" + re.escape(term) + r"(?![a-z0-9+#])"
        if re.search(pattern, t):
            found.add(canon)
    return found


# ---------------------------------------------------------------
# Step 4: split the resume into sections (rule-based)
# ---------------------------------------------------------------
SECTION_HEADS = {
    "skills": ["skills", "technical skills", "key skills"],
    "education": ["education", "academic background", "qualifications"],
    "experience": ["experience", "work experience", "professional experience", "internship", "internships"],
    "certifications": ["certifications", "certificates", "courses"],
    "projects": ["projects", "academic projects", "personal projects"],
}


def split_sections(text):
    """Group lines under the most recent heading they appear after."""
    sections = {k: [] for k in SECTION_HEADS}
    current = None
    for line in text.split("\n"):
        key = line.strip().lower().rstrip(":")
        found = next((s for s, names in SECTION_HEADS.items() if key in names), None)
        if found:
            current = found                      # a heading line: switch section
        elif current:
            sections[current].append(line.strip())   # a normal line: belongs to current section
    return {k: " ".join(v).strip() for k, v in sections.items()}


# ---------------------------------------------------------------
# Step 5: full pipeline for one resume
# ---------------------------------------------------------------
def parse_resume(path):
    """Never crashes: on a bad file it returns a dict with an 'error' message instead."""
    result = {"file": os.path.basename(path), "error": None}
    try:
        raw = extract_text(path)
    except Exception as e:
        result["error"] = str(e)
        return result

    text = clean_text(raw)
    sec = split_sections(text)
    lines = [l for l in text.split("\n") if l.strip()]

    # years of experience: first "<number> years" pattern in the text
    yrs = re.search(r"(\d+)\+?\s*(?:years|yrs|year)", text.lower())
    result.update({
        "name": lines[0] if lines else "Unknown",   # display only - NEVER used in scoring
        "text": text,
        "skills": sorted(extract_skills(text)),
        "education": sec["education"] or "Not found",
        "experience": sec["experience"] or "Not found",
        "certifications": sec["certifications"] or "Not found",
        "projects": sec["projects"] or "Not found",
        "years_experience": int(yrs.group(1)) if yrs else None,
    })
    return result
