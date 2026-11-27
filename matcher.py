"""
matcher.py  -  MODULE 2: job-description matching and explainable scoring
Final score = 70% skill match + 30% text similarity. No name/age/gender/etc. is ever used.
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from extractor import extract_skills, ALIAS_TO_CANON

# ---------------- scoring settings (documented in README) ----------------
W_REQUIRED, W_OPTIONAL, W_EXTRA = 1.0, 0.5, 0.5   # weight of each kind of keyword
SKILL_PART, SIM_PART = 0.7, 0.3                    # final = 70% skills + 30% text similarity

# words that appear in almost every JD and are not skills
GENERIC = {"looking", "required", "requires", "experience", "candidates", "candidate", "strong", "skills", "skill",
           "knowledge", "good", "expected", "plus", "hiring", "job", "description", "bachelor", "degree", "related",
           "field", "years", "year", "work", "working", "ability", "team", "role", "responsibilities", "preferred",
           "must", "including", "etc", "excellent", "able", "using", "use", "new", "company", "need", "needs", "seeking",
           "proficiency", "understanding", "familiarity", "hands", "science", "computer",
           "scientist", "developer", "stack", "skilled", "full", "engineer", "analyst", "web", "data", "senior",
           "junior", "intern", "tech", "technical"}

# sentences containing these words mark their skills as "optional"
OPTIONAL_CUES = ["plus", "preferred", "nice to have", "good to have", "bonus", "desirable", "advantage"]


def split_sentences(text):
    return [s.strip() for s in re.split(r"\n|(?<=[a-zA-Z0-9])\.\s+", text) if s.strip()]


# ---------------------------------------------------------------
# Step 1: extract keywords from ANY job description
# ---------------------------------------------------------------
def extract_jd_keywords(jd_text, corpus_texts, top_n=5):
    """Layer 1: dictionary skills (required vs optional). Layer 2: TF-IDF phrases for anything new."""
    required, optional = set(), set()
    for sent in split_sentences(jd_text):
        found = extract_skills(sent)
        if any(cue in sent.lower() for cue in OPTIONAL_CUES):
            optional |= found
        else:
            required |= found
    optional -= required          # a skill that is required anywhere stays required

    # TF-IDF: JD vs resumes. Words frequent in the JD but rare in resumes score high.
    covered_words = set(w for term in ALIAS_TO_CANON for w in re.findall(r"[a-z0-9]+", term))
    stops = list(ENGLISH_STOP_WORDS.union(GENERIC))
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=stops)
    X = vec.fit_transform([jd_text.lower()] + list(corpus_texts))
    scores = X[0].toarray().ravel()
    terms = vec.get_feature_names_out()

    extras = []
    for idx in scores.argsort()[::-1]:
        if scores[idx] <= 0 or len(extras) >= top_n:
            break
        term = terms[idx]
        if any(w in covered_words for w in term.split()):
            continue                                    # already covered by the dictionary
        if any(term in e or e in term for e in extras):
            continue                                    # skip near-duplicates
        extras.append(term)
    return {"required": sorted(required), "optional": sorted(optional), "extra": extras}


# ---------------------------------------------------------------
# Step 2: score one resume against the JD keywords
# ---------------------------------------------------------------
def anonymize(text):
    """Remove the name (first line) and emails so identity cannot influence text similarity."""
    lines = text.split("\n")[1:]
    return re.sub(r"\S+@\S+", " ", " ".join(lines))


def match_resume(resume, kw, jd_text):
    r_text = resume["text"].lower()
    r_skills = set(resume["skills"])

    items = [(s, W_REQUIRED, "required") for s in kw["required"]] + \
            [(s, W_OPTIONAL, "optional") for s in kw["optional"]] + \
            [(s, W_EXTRA, "extra") for s in kw["extra"]]

    matched, missing, got, total = [], [], 0.0, 0.0
    for term, w, kind in items:
        if kind == "extra":
            hit = bool(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", r_text))
        else:
            hit = term in r_skills
        total += w
        if hit:
            matched.append(term)
            got += w
        else:
            missing.append(term)

    skill_score = 100 * got / total if total else 0.0       # weighted % of keywords found

    # cosine similarity between the JD and the anonymized resume (0-100%)
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    M = vec.fit_transform([jd_text.lower(), anonymize(resume["text"]).lower()])
    sim = float(cosine_similarity(M[0], M[1])[0][0]) * 100

    final = SKILL_PART * skill_score + SIM_PART * sim
    n_req = len(kw["required"])
    req_hit = len([m for m in matched if m in kw["required"]])
    reason = (f"Matched {req_hit}/{n_req} required skills; skill score {skill_score:.0f}%, "
              f"text similarity {sim:.0f}% -> final = 0.7*{skill_score:.0f} + 0.3*{sim:.0f} = {final:.1f}")

    return {"file": resume["file"], "name": resume["name"], "final_score": round(final, 1),
            "skill_score": round(skill_score, 1), "similarity": round(sim, 1),
            "matched": matched, "missing": missing, "years": resume["years_experience"], "reason": reason}


# ---------------------------------------------------------------
# Step 3: rank all valid resumes
# ---------------------------------------------------------------
def rank_candidates(parsed, jd_text):
    valid = [p for p in parsed if not p["error"]]      # bad files are skipped, not fatal
    kw = extract_jd_keywords(jd_text, [p["text"] for p in valid])
    df = pd.DataFrame([match_resume(p, kw, jd_text) for p in valid])
    return kw, df.sort_values("final_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# Sorting options shared by the notebook dashboard and the web app
# ---------------------------------------------------------------
SORT_OPTIONS = {
    "Score: high to low": ("final_score", False),
    "Score: low to high": ("final_score", True),
    "Years of experience: high to low": ("years", False),
    "Name: A to Z": ("name", True),
}


def sort_ranked(ranked, option):
    col, ascending = SORT_OPTIONS[option]
    return ranked.sort_values(col, ascending=ascending, na_position="last", kind="stable")
