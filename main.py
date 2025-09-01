import streamlit as st
import fitz  # PyMuPDF
import re
import pytesseract
from PIL import Image
import io
import requests
import spacy
from collections import Counter

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Hugging Face API
HUGGINGFACE_TOKEN = "" #use your huggingface token
MODEL_URL = "https://api-inference.huggingface.co/models/Nucha/Nucha_ITSkillNER_BERT"
NAME_NER_MODEL = "https://api-inference.huggingface.co/models/dbmdz/bert-large-cased-finetuned-conll03-english"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

# Define skills
SOFT_SKILLS = [
    'communication', 'leadership', 'teamwork', 'adaptability',
    'problem-solving', 'creativity', 'time management', 'critical thinking'
]
TECHNICAL_SKILLS = [
    'python', 'java', 'sql', 'html', 'css', 'javascript', 'machine learning',
    'deep learning', 'tensorflow', 'pytorch', 'data analysis', 'data science'
]

COURSE_RECOMMENDATIONS = {
    'python': 'https://www.coursera.org/specializations/python',
    'java': 'https://www.udemy.com/course/java-the-complete-java-developer-course/',
    'machine learning': 'https://www.coursera.org/learn/machine-learning',
    'data analysis': 'https://www.datacamp.com/tracks/data-analyst-with-python',
    'sql': 'https://www.codecademy.com/learn/learn-sql',
    'communication': 'https://www.udemy.com/course/communication-skills/',
    'teamwork': 'https://www.futurelearn.com/courses/teamwork-skills',
    'problem-solving': 'https://www.linkedin.com/learning/problem-solving-techniques',
    'leadership': 'https://www.edx.org/course/leadership-training',
    'adaptability': 'https://www.udemy.com/course/adaptability-skills/',
    'creativity': 'https://www.linkedin.com/learning/creativity-for-all',
    'time management': 'https://www.coursera.org/learn/work-smarter-not-harder',
    'critical thinking': 'https://www.edx.org/course/critical-thinking'
}

JOB_SUGGESTIONS = {
    'machine learning': 'Machine Learning Engineer',
    'data analysis': 'Data Analyst',
    'python': 'Backend Developer',
    'java': 'Java Developer',
    'sql': 'Database Engineer',
    'html': 'Frontend Developer',
    'css': 'Frontend Developer',
    'javascript': 'Full Stack Developer',
    'tensorflow': 'Deep Learning Engineer',
    'pytorch': 'AI Researcher'
}

# PDF Text Extraction
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# Hugging Face-based name extractor using offset-based grouping
def extract_name_with_ner(text):
    response = requests.post(NAME_NER_MODEL, headers=headers, json={"inputs": text[:1000]})
    if response.status_code == 200:
        results = response.json()
        names = []
        current_name = ""
        for ent in results:
            if ent['entity_group'] == 'PER':
                if ent['word'].startswith("##"):
                    current_name += ent['word'][2:]
                else:
                    if current_name:
                        names.append(current_name.strip().title())
                    current_name = ent['word']
        if current_name:
            names.append(current_name.strip().title())
        most_common = Counter(names).most_common(1)
        return most_common[0][0] if most_common else "Not found"
    return "Not found"


# Best-effort name extraction with reordered fallbacks

def extract_best_name(uploaded_file, text):
    candidates = []

    # Step 1: Hugging Face NER first (offset merging for robustness)
    try:
        response = requests.post(NAME_NER_MODEL, headers=headers, json={"inputs": text[:1000]})
        if response.status_code == 200:
            results = response.json()
            current = ""
            names = []
            for ent in results:
                if ent['entity_group'] == 'PER':
                    if ent['word'].startswith("##"):
                        current += ent['word'][2:]
                    else:
                        if current:
                            names.append(current.strip().title())
                        current = ent['word']
            if current:
                names.append(current.strip().title())
            candidates.extend(names)
    except Exception:
        pass

    # Step 2: Top lines of raw text for PERSON entities
    try:
        top_lines = "\n".join(text.splitlines()[:10])
        doc_top = nlp(top_lines)
        top_names = [ent.text.strip().title() for ent in doc_top.ents if ent.label_ == "PERSON" and len(ent.text.strip().split()) <= 4 and '@' not in ent.text]
        candidates.extend(top_names)
    except Exception:
        pass

    # Step 3: OCR-based fallback scanning all pages
    try:
        uploaded_file.seek(0)
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
                page_text = pytesseract.image_to_string(image)
                doc_ocr = nlp(page_text)
                page_names = [ent.text.strip().title() for ent in doc_ocr.ents if ent.label_ == "PERSON" and len(ent.text.strip().split()) <= 4 and '@' not in ent.text]
                candidates.extend(page_names)
    except Exception:
        pass

    return Counter(candidates).most_common(1)[0][0] if candidates else "Not found"

# Contact Info
def extract_contact_info(text):
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    phone = re.findall(r'\+?\d[\d\-\s]{8,}\d', text)
    return email[0] if email else "Not found", phone[0] if phone else "Not found"

# Hugging Face NER
def call_hf_skill_ner(text):
    response = requests.post(MODEL_URL, headers=headers, json={"inputs": text[:4000]})
    if response.status_code == 200:
        return response.json()
    else:
        return []

# Parse skills from NER tags
def parse_skills_from_ner(ner_results):
    skills = []
    current_skill = ""
    current_label = ""

    for ent in ner_results:
        word = ent.get("word", "").replace("##", "")
        label = ent.get("entity", "")

        if label.startswith("B-"):
            if current_skill:
                skills.append((current_skill.strip().lower(), current_label))
            current_skill = word
            current_label = label.split("-")[1]
        elif label.startswith("I-") and current_label:
            current_skill += " " + word
        else:
            if current_skill:
                skills.append((current_skill.strip().lower(), current_label))
            current_skill = ""
            current_label = ""

    if current_skill:
        skills.append((current_skill.strip().lower(), current_label))
    return skills

# Backup keyword skill extractor using spaCy
def fallback_skill_extraction(text):
    doc = nlp(text.lower())
    tokens = set([token.text for token in doc if not token.is_stop])
    technical = [skill for skill in TECHNICAL_SKILLS if skill in tokens]
    soft = [skill for skill in SOFT_SKILLS if skill in tokens]
    return list(set(technical)), list(set(soft))

# Main skill extractor
def extract_skills(text):
    try:
        ner_results = call_hf_skill_ner(text)
        parsed = parse_skills_from_ner(ner_results)
        tech = [s for s, label in parsed if label == 'HSKILL']
        soft = [s for s, label in parsed if label == 'SSKILL']
    except Exception:
        tech, soft = [], []

    if not tech and not soft:
        tech, soft = fallback_skill_extraction(text)
    return list(set(tech)), list(set(soft))


def calculate_ats_score(text):
    tech, soft = extract_skills(text)
    tech_score = (len(tech) / 10) * 85
    soft_score = (len(soft) / len(SOFT_SKILLS)) * 15
    return min(round(tech_score + soft_score), 100)


def recommend_courses(missing_skills):
    return {skill: COURSE_RECOMMENDATIONS[skill] for skill in missing_skills if skill in COURSE_RECOMMENDATIONS}


def predict_jobs(skills):
    return list({JOB_SUGGESTIONS[skill] for skill in skills if skill in JOB_SUGGESTIONS})


# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer with ATS Score, Skills, Job Prediction & Course Suggestions")

menu = st.sidebar.radio("Navigate", ["Upload Resume", "View Score", "Extracted Info", "Course Suggestions", "Job Predictor"])

if menu == "Upload Resume":
    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.session_state['resume_text'] = resume_text
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['name'] = extract_best_name(uploaded_file, resume_text)
        st.balloons()
        st.success("Resume uploaded and processed successfully!")

elif menu == "View Score":
    if 'resume_text' not in st.session_state:
        st.warning("Please upload a resume first.")
    else:
        ats_score = calculate_ats_score(st.session_state['resume_text'])
        st.subheader("✅ ATS Resume Score")
        st.progress(ats_score)
        st.success(f"Your ATS score is: {ats_score} / 100")

elif menu == "Extracted Info":
    if 'resume_text' not in st.session_state:
        st.warning("Please upload a resume first.")
    else:
        name = st.text_input("👤 Name", value=st.session_state.get("name", "Not found"))
        email, phone = extract_contact_info(st.session_state['resume_text'])
        tech_skills, soft_skills = extract_skills(st.session_state['resume_text'])

        st.subheader("👤 Contact Information")
        st.write(f"**Name:** {name}")
        st.write(f"**Email:** {email}")
        st.write(f"**Phone:** {phone}")

        st.subheader("📊 Extracted Skills")
        st.markdown("**Technical Skills:**")
        st.write(", ".join(tech_skills) if tech_skills else "No technical skills identified.")

        st.markdown("**Soft Skills:**")
        st.write(", ".join(soft_skills) if soft_skills else "No soft skills identified.")

elif menu == "Course Suggestions":
    if 'resume_text' not in st.session_state:
        st.warning("Please upload a resume first.")
    else:
        tech_skills, soft_skills = extract_skills(st.session_state['resume_text'])
        user_skills = set(tech_skills + soft_skills)
        all_skills = set(list(JOB_SUGGESTIONS.keys()) + SOFT_SKILLS)
        missing_skills = list(all_skills - user_skills)

        recommendations = recommend_courses(missing_skills)

        st.subheader("📚 Recommended Courses Based on Missing Skills")
        if recommendations:
            for skill, url in recommendations.items():
                st.markdown(f"- [{skill.title()}]({url})")
        else:
            st.info("🎉 You're already skilled across all key areas!")

elif menu == "Job Predictor":
    if 'resume_text' not in st.session_state:
        st.warning("Please upload a resume first.")
    else:
        tech_skills, soft_skills = extract_skills(st.session_state['resume_text'])
        all_skills = tech_skills + soft_skills
        suggested_jobs = predict_jobs(all_skills)

        st.subheader("🧑‍💼 Suggested Job Roles Based on Your Skills")
        if suggested_jobs:
            for job in suggested_jobs:
                st.markdown(f"- {job}")
        else:
            st.info("No matching job roles found. Try adding more technical keywords.")
