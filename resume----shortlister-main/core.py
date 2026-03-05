import requests
from docx import Document
import os
import re
import json
import hashlib
from datetime import datetime
import math
from email_sender import EmailSender
import PyPDF2

# Ollama configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


def _extract_outermost_json(text):
    """
    Robustly extract the first complete top-level JSON object from text.
    Handles nested objects and arrays correctly by counting braces.
    """
    # Strip code-fence markers
    text = re.sub(r'```[a-zA-Z]*', '', text).strip()

    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    raise ValueError("Incomplete JSON object in response (unbalanced braces)")


# DataProcessor Class
class DataProcessor:
    """Unified data processor for resume shortlisting system."""

    # --- 1. Utility/Helper Methods ---
    def __init__(self, model_name=None):
        self.model_name = model_name or OLLAMA_MODEL
        self.ollama_url = OLLAMA_BASE_URL
        self.supported_extensions = {'.pdf', '.docx'}
        self.email_sender = EmailSender(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="lokeshboggula06@gmail.com",
            sender_password="Lokesh@2006"
        )

    def log(self, message, level="DEBUG"):
        print(f"[{level}] {message}")

    def execute_ai_operation(self, prompt, operation_name, fallback_func=None, fallback_args=None):
        """Send prompt to local Ollama LLaMA 3.1 and parse the JSON response."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"        # Constrain Ollama output to JSON
            }
            self.log(f"{operation_name}: Sending request to Ollama ({self.model_name}) ...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=180             # 3-minute timeout
            )
            response.raise_for_status()
            ollama_data = response.json()
            response_text = ollama_data.get("response", "")
            self.log(f"{operation_name} RAW response: {response_text}")

            # Use brace-counting extractor — correctly handles nested objects/arrays
            json_text = _extract_outermost_json(response_text)
            parsed_data = json.loads(json_text)
            self.log(f"{operation_name} PARSED data: {parsed_data}")
            return parsed_data

        except requests.exceptions.ConnectionError:
            self.log(f"Could not connect to Ollama at {self.ollama_url}. Is Ollama running?", "ERROR")
            if fallback_func and fallback_args is not None:
                return fallback_func(*fallback_args)
            raise
        except requests.exceptions.Timeout:
            self.log(f"{operation_name}: Ollama request timed out. Using fallback.", "ERROR")
            if fallback_func and fallback_args is not None:
                return fallback_func(*fallback_args)
            raise
        except Exception as e:
            self.log(f"Error during {operation_name}: {e}. Using fallback.")
            if fallback_func and fallback_args is not None:
                return fallback_func(*fallback_args)
            else:
                raise e

    def normalize_score(self, score):
        return float(score * 100 if score <= 1.0 else score)

    def create_fallback_data(self, data_type="evaluation", **kwargs):
        if data_type == "evaluation":
            basic_score = float(kwargs.get("basic_score", 0))
            return {
                "overall_score": basic_score,
                "technical_skills_score": basic_score,
                "experience_score": basic_score,
                "education_score": basic_score,
                "soft_skills_score": basic_score,
                "detailed_feedback": "Basic word overlap analysis (AI evaluation failed)",
                "strengths": ["Basic keyword matching"],
                "areas_for_improvement": ["AI evaluation unavailable"],
                "recommendation": "MAYBE" if basic_score >= 50 else "REJECT"
            }
        elif data_type == "ner":
            return {
                "name": kwargs.get("name", "N/A"),
                "email": kwargs.get("email", "N/A"),
                "phone": kwargs.get("phone", "N/A"),
            }
        return {}

    def parse_json_response(self, response_text, function_name="Unknown"):
        try:
            json_text = _extract_outermost_json(response_text)
            parsed_data = json.loads(json_text)
            if isinstance(parsed_data, list):
                parsed_data = parsed_data[0] if parsed_data else {}
            self.log(f"Successfully parsed {function_name} data: {parsed_data}")
            return parsed_data
        except Exception as e:
            self.log(f"JSON parsing failed for {function_name}: {e}")
            raise ValueError(f"JSON parsing failed for {function_name}")

    def safe_str(self, val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return ""
        return str(val)

    # --- 2. File/Data Extraction ---
    def validate_file_type(self, file_path):
        return os.path.splitext(file_path)[1].lower() in self.supported_extensions

    def extract_file_content(self, file_path):
        try:
            if not self.validate_file_type(file_path):
                self.log(f"Unsupported file type: {os.path.splitext(file_path)[1].lower()}")
                return []
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                blocks = []
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            lines = [line.strip() for line in text.split('\n') if line.strip()]
                            blocks.extend(lines)
                return blocks
            elif ext == ".docx":
                doc = Document(file_path)
                return [para.text for para in doc.paragraphs if para.text.strip()]
        except Exception as e:
            self.log(f"Error extracting content from {file_path}: {e}")
            return []

    def load_job_data(self, filename="job_descriptions/all_jobs.json"):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.log(f"Could not load job data: {e}")
            return {}

    def get_job_info(self, role=None, title=None, action="load"):
        all_jobs = self.load_job_data()
        if action == "load" and role:
            role_key = role.lower().replace(' ', '_') if ' ' in role else role
            return all_jobs.get(role_key, {})
        elif action == "names":
            return [job_data["title"] for job_data in all_jobs.values() if isinstance(job_data, dict) and "title" in job_data]
        elif action == "key" and title:
            for key, job_data in all_jobs.items():
                if isinstance(job_data, dict) and job_data.get("title") == title:
                    return key
        return None

    def extract_job_description_text(self, jd_data):
        if isinstance(jd_data, dict):
            return " ".join([
                jd_data.get("title", ""),
                jd_data.get("summary", ""),
                " ".join(jd_data.get("responsibilities", [])),
                " ".join(jd_data.get("qualifications", []))
            ])
        return str(jd_data)

    # --- 3. AI Operations ---
    def compute_similarity(self, resume_text, job_description):
        prompt = f'''You are an expert HR recruiter and technical interviewer with 15+ years of hiring experience.
Your task is to rigorously evaluate a candidate resume against a specific job description and return a precise, differentiated score.

SCORING RULES (read carefully):
- Scores must be floating-point numbers between 0.0 and 100.0 (e.g. 67.35, 81.20, 44.75)
- Do NOT use round numbers like 70, 80, 90 — use decimals that reflect precise fit
- Score 90+ only if the resume is an exceptional, near-perfect match
- Score 70-89 for good matches with minor gaps
- Score 50-69 for partial matches with notable gaps  
- Score below 50 for poor matches

SCORING CRITERIA:
1. technical_skills_score (40% weight):
   - Exact keyword/technology matches between resume and JD requirements
   - Depth of expertise in listed skills (years, project complexity)
   - Missing critical technologies heavily penalize this score

2. experience_score (30% weight):
   - Years of relevant experience vs JD requirement
   - Relevance of past roles, projects, and industry
   - Measurable achievements and impact

3. education_score (15% weight):
   - Degree level and field vs JD requirements
   - Certifications, courses, or training relevant to the role
   - Academic or professional credentials

4. soft_skills_score (15% weight):
   - Evidence of communication, leadership, collaboration
   - Problem-solving examples in resume
   - Cultural and role-fit indicators

OVERALL SCORE = (technical * 0.40) + (experience * 0.30) + (education * 0.15) + (soft_skills * 0.15)
Compute it yourself mathematically — do not guess.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Respond ONLY with this exact JSON structure (no extra text, no markdown):
{{
    "overall_score": <computed weighted float, e.g. 67.35>,
    "technical_skills_score": <float 0.0-100.0>,
    "experience_score": <float 0.0-100.0>,
    "education_score": <float 0.0-100.0>,
    "soft_skills_score": <float 0.0-100.0>,
    "detailed_feedback": "<2-3 sentence honest assessment explaining the score>",
    "strengths": ["<specific strength from resume>", "<specific strength>", "<specific strength>"],
    "areas_for_improvement": ["<specific gap vs JD>", "<specific gap>", "<specific gap>"],
    "recommendation": "<RECOMMEND if overall>=75 | MAYBE if 50<=overall<75 | REJECT if overall<50>"
}}'''

        def basic_word_overlap_fallback():
            job_words = set(job_description.lower().split())
            resume_words = set(resume_text.lower().split())
            match_count = len(job_words & resume_words)
            total_job_words = len(job_words)
            basic_score = (match_count / total_job_words) * 100 if total_job_words > 0 else 0
            if match_count > 0:
                match_ratio = match_count / total_job_words
                if match_ratio > 0.3:
                    basic_score = min(100, basic_score * 1.2)
                elif match_ratio < 0.1:
                    basic_score = max(0, basic_score * 0.8)
                if basic_score < 10:
                    basic_score = 10.0
            self.log(f"Fallback word overlap: {match_count}/{total_job_words} words, score: {basic_score:.2f}")
            return self.create_fallback_data("evaluation", basic_score=basic_score)

        return self.execute_ai_operation(
            prompt=prompt,
            operation_name="AI evaluation",
            fallback_func=basic_word_overlap_fallback,
            fallback_args=()
        )

    def fallback_ner_extraction(self, text):
        """Regex-based NER fallback when LLaMA is unavailable."""
        name_patterns = [
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})",   # Title case names
            r"([A-Z][A-Z]+\s[A-Z][A-Z]+)",             # ALL CAPS names
        ]
        name = "N/A"
        skip_words = {
            'machine learning', 'data science', 'artificial intelligence',
            'deep learning', 'computer science', 'software engineer',
            'business analyst', 'project manager', 'curriculum vitae',
            'resume', 'objective', 'summary', 'experience', 'education',
            'skills', 'languages', 'contact', 'references'
        }
        for pattern in name_patterns:
            for match in re.finditer(pattern, text):
                candidate = match.group(1)
                if candidate.lower() not in skip_words and len(candidate) > 4:
                    name = candidate
                    break
            if name != "N/A":
                break

        if name == "N/A":
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                name = lines[0][:60]  # First line, max 60 chars

        email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
        phone_match = re.search(r"(\+?\d[\d\s\-\(\)]{7,15}\d)", text)
        email = email_match.group(0) if email_match else None

        if not email:
            base = (name or "N/A") + text[:100]
            email = f"{hashlib.md5(base.encode()).hexdigest()[:8]}@noemail.local"

        self.log(f"[FALLBACK NER] name: {name}, email: {email}")
        return self.create_fallback_data("ner",
            name=name,
            email=email,
            phone=phone_match.group(0).strip() if phone_match else "N/A"
        )

    def extract_ner(self, text, role=None):
        """Extract name, email, and phone from resume text using LLaMA 3.1."""
        # Limit context to first 3000 chars — header area has the contact info
        context_text = text[:3000]

        prompt = f'''You are a precise resume parser. Extract ONLY these three fields from the resume text below.

EXTRACTION RULES:
- name: The candidate's FULL PERSONAL NAME (first + last, possibly middle). 
  * It appears at the TOP of the resume, often the largest text or first line.
  * It is a PERSON'S name — NOT a company, university, degree, city, or job title.
  * Examples of valid names: "Rahul Sharma", "John Michael Smith", "Priya Reddy"
  * If truly not found, use "N/A"
- email: A valid email address (format: user@domain.tld).
  * Look near phone/contact sections. Take the FIRST real email found.
  * If not found, use "N/A"
- phone: A phone/mobile number (digits, may have +, spaces, dashes).
  * If not found, use "N/A"

RESUME TEXT (first section — most likely to contain contact info):
{context_text}

Respond ONLY with this exact JSON (no extra text, no markdown fences):
{{"name": "<full name or N/A>", "email": "<email or N/A>", "phone": "<phone or N/A>"}}'''

        return self.execute_ai_operation(
            prompt=prompt,
            operation_name="NER extraction",
            fallback_func=self.fallback_ner_extraction,
            fallback_args=(text,)
        )

    # --- 4. Result Construction ---
    def build_result_dict(self, ner_data, evaluation_data, overall_score, technical_score,
                         experience_score, education_score, soft_skills_score,
                         recommendation, message_suffix=""):
        overall_score = self.normalize_score(overall_score)
        technical_score = self.normalize_score(technical_score)
        experience_score = self.normalize_score(experience_score)
        education_score = self.normalize_score(education_score)
        soft_skills_score = self.normalize_score(soft_skills_score)

        result_str = (
            f"✅ Processed {ner_data['name']} ({ner_data['email']}) - "
            f"Overall: {overall_score:.2f}% | Tech: {technical_score:.2f}% | "
            f"Exp: {experience_score:.2f}% | Edu: {education_score:.2f}% | "
            f"Soft: {soft_skills_score:.2f}% | Rec: {recommendation}{message_suffix}"
        )
        return {
            "result_str": result_str,
            "overall_score": overall_score,
            "technical_score": technical_score,
            "experience_score": experience_score,
            "education_score": education_score,
            "soft_skills_score": soft_skills_score,
            "recommendation": recommendation,
            "ner_data": ner_data,
            "evaluation_data": evaluation_data
        }

    def create_error_result(self, error_message, error_type="General"):
        return {
            "result_str": f"❌ {error_type} error: {error_message}",
            "overall_score": 0.0,
            "technical_score": 0.0,
            "experience_score": 0.0,
            "education_score": 0.0,
            "soft_skills_score": 0.0,
            "recommendation": "REJECT",
            "ner_data": {"name": "N/A", "email": "N/A", "phone": "N/A"},
            "evaluation_data": {}
        }

    def send_shortlist_email(self, recipient_email, name, role, match_percentage):
        return self.email_sender.send_email(
            recipient_email=recipient_email,
            name=name,
            role=role,
            match_percentage=match_percentage
        )

    # --- 5. Excel/Historical Data Management ---
    def manage_data(self, action="save", **kwargs):
        try:
            import pandas as pd
            if action == "save" and all(k in kwargs for k in ["results_data", "role", "threshold"]):
                results_dir = os.path.join(os.getcwd(), "results")
                os.makedirs(results_dir, exist_ok=True)
                excel_file = os.path.join(results_dir, "ranked_resume_results.xlsx")
                records = []
                for result in kwargs["results_data"]:
                    if isinstance(result, dict) and "result_str" in result:
                        candidate_name = result.get("ner_data", {}).get("name", "N/A")
                        email = result.get("ner_data", {}).get("email", "N/A")
                        overall_score = float(result.get("overall_score", 0))
                        records.append({
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Job Role': kwargs["role"],
                            'Threshold': kwargs["threshold"],
                            'Candidate Name': candidate_name,
                            'Email': email,
                            'Overall Score (%)': overall_score,
                            'Technical Skills (%)': float(result.get("technical_score", 0)),
                            'Experience (%)': float(result.get("experience_score", 0)),
                            'Education (%)': float(result.get("education_score", 0)),
                            'Soft Skills (%)': float(result.get("soft_skills_score", 0)),
                            'Recommendation': result.get("recommendation", "MAYBE"),
                            'Status': "Ready for Email" if (overall_score >= kwargs["threshold"] and email not in ("N/A", "")) else "Below Threshold",
                            'Detailed Feedback': result.get("evaluation_data", {}).get("detailed_feedback", "N/A"),
                            'Strengths': "; ".join(result.get("evaluation_data", {}).get("strengths", [])),
                            'Areas for Improvement': "; ".join(result.get("evaluation_data", {}).get("areas_for_improvement", [])),
                            'Full Result': result["result_str"]
                        })
                    else:
                        records.append({
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Job Role': kwargs["role"],
                            'Threshold': kwargs["threshold"],
                            'Candidate Name': 'N/A',
                            'Email': 'N/A',
                            'Overall Score (%)': 0.0,
                            'Technical Skills (%)': 0.0,
                            'Experience (%)': 0.0,
                            'Education (%)': 0.0,
                            'Soft Skills (%)': 0.0,
                            'Recommendation': 'REJECT',
                            'Status': 'Error',
                            'Detailed Feedback': 'N/A',
                            'Strengths': 'N/A',
                            'Areas for Improvement': 'N/A',
                            'Full Result': str(result) if isinstance(result, str) else 'Error processing'
                        })
                df = pd.DataFrame(records)
                # Percentile as precise float (0.0–100.0)
                if not df.empty:
                    df['Percentile'] = df['Overall Score (%)'].rank(pct=True, method='max') * 100
                else:
                    df['Percentile'] = 0.0
                df = df.sort_values('Overall Score (%)', ascending=False).reset_index(drop=True)
                df['Rank'] = range(1, len(df) + 1)
                df.to_excel(excel_file, index=False)
                self.log(f"Ranked results saved to Excel: {excel_file}")
                return excel_file

            elif action == "load" and "role" in kwargs:
                results_dir = os.path.join(os.getcwd(), "results")
                if not os.path.exists(results_dir):
                    return []
                all_results = []
                for filename in os.listdir(results_dir):
                    if filename.endswith('.xlsx') and 'ranked_resume_results' in filename:
                        file_path = os.path.join(results_dir, filename)
                        try:
                            df = pd.read_excel(file_path)
                            role_results = df[df['Job Role'] == kwargs["role"]]
                            for _, row in role_results.iterrows():
                                result_data = {
                                    'result_str': row.get('Full Result', ''),
                                    'overall_score': float(row.get('Overall Score (%)', 0) or 0),
                                    'technical_score': float(row.get('Technical Skills (%)', 0) or 0),
                                    'experience_score': float(row.get('Experience (%)', 0) or 0),
                                    'education_score': float(row.get('Education (%)', 0) or 0),
                                    'soft_skills_score': float(row.get('Soft Skills (%)', 0) or 0),
                                    'recommendation': row.get('Recommendation', 'MAYBE'),
                                    'ner_data': {
                                        'name': self.safe_str(row.get('Candidate Name', 'N/A')) or 'N/A',
                                        'email': self.safe_str(row.get('Email', 'N/A')),
                                        'phone': 'N/A'
                                    },
                                    'evaluation_data': {
                                        'detailed_feedback': row.get('Detailed Feedback', 'N/A'),
                                        'strengths': (row.get('Strengths') or '').split('; '),
                                        'areas_for_improvement': (row.get('Areas for Improvement') or '').split('; ')
                                    },
                                    'timestamp': row.get('Timestamp', ''),
                                    'batch_id': filename
                                }
                                all_results.append(result_data)
                        except Exception as e:
                            self.log(f"Error reading Excel file {filename}: {e}")
                            continue
                seen_keys = set()
                deduped_results = []
                for result in all_results:
                    name = str(result.get('ner_data', {}).get('name', '')).lower()
                    email = str(result.get('ner_data', {}).get('email', '')).lower()
                    key = (name, email)
                    if key not in seen_keys:
                        deduped_results.append(result)
                        seen_keys.add(key)
                self.log(f"Deduplicated results: {len(deduped_results)} out of {len(all_results)}")
                return deduped_results

            elif action == "rank" and "all_results" in kwargs:
                top_n = kwargs.get("top_n", 5)
                sorted_results = sorted(kwargs["all_results"], key=lambda x: x.get('overall_score', 0), reverse=True)
                top_results = sorted_results[:top_n]
                for i, result in enumerate(top_results, 1):
                    result['global_rank'] = i
                    result['total_candidates'] = len(kwargs["all_results"])
                return top_results
            return []
        except Exception as e:
            self.log(f"Error in data operation: {e}")
            return []

    # --- 6. Main Orchestration ---
    def process_uploaded_resume(self, file_path, role, threshold):
        full_text = ""
        try:
            self.log(f"Starting resume processing: {file_path}, role: {role}, threshold: {threshold}")

            if not os.path.exists(file_path):
                return self.create_error_result("File not found.", "File Not Found")

            blocks = self.extract_file_content(file_path)
            if not blocks:
                return self.create_error_result("No content could be extracted from the file.", "Content Extraction")

            full_text = " ".join(blocks)
            self.log(f"Extracted {len(blocks)} blocks, {len(full_text)} chars")

            jd_data = self.get_job_info(role=role, action="load")
            if not jd_data:
                return self.create_error_result(f"Unknown job role: {role}", "Job Title")
            jd_text = self.extract_job_description_text(jd_data)

            # === NER first (fast) ===
            ner_data = self.extract_ner(full_text, role=role)
            self.log(f"NER result: {ner_data}")

            # === Scoring (slower, LLaMA 3.1) ===
            evaluation_data = self.compute_similarity(full_text, jd_text)
            self.log(f"Evaluation result: {evaluation_data}")

            scores = {
                "overall": float(evaluation_data.get("overall_score", 0)),
                "technical": float(evaluation_data.get("technical_skills_score", 0)),
                "experience": float(evaluation_data.get("experience_score", 0)),
                "education": float(evaluation_data.get("education_score", 0)),
                "soft_skills": float(evaluation_data.get("soft_skills_score", 0))
            }
            recommendation = evaluation_data.get("recommendation", "MAYBE")

            if scores["overall"] >= threshold and ner_data.get("email") not in (None, "N/A", ""):
                message_suffix = " (Ready to send email)"
            else:
                message_suffix = " (Below threshold or no valid email)"

            return self.build_result_dict(
                ner_data=ner_data,
                evaluation_data=evaluation_data,
                overall_score=scores["overall"],
                technical_score=scores["technical"],
                experience_score=scores["experience"],
                education_score=scores["education"],
                soft_skills_score=scores["soft_skills"],
                recommendation=recommendation,
                message_suffix=message_suffix
            )

        except FileNotFoundError:
            return self.create_error_result("File not found.", "File Not Found")
        except PermissionError:
            return self.create_error_result("Permission denied accessing the file.", "Permission")
        except Exception as e:
            self.log(f"Processing error: {e}")
            try:
                fallback_evaluation = self.create_fallback_data("evaluation", basic_score=50.0)
                fallback_ner = self.fallback_ner_extraction(full_text) if full_text else {"name": "N/A", "email": "N/A", "phone": "N/A"}
                return self.build_result_dict(
                    ner_data=fallback_ner,
                    evaluation_data=fallback_evaluation,
                    overall_score=fallback_evaluation["overall_score"],
                    technical_score=fallback_evaluation["technical_skills_score"],
                    experience_score=fallback_evaluation["experience_score"],
                    education_score=fallback_evaluation["education_score"],
                    soft_skills_score=fallback_evaluation["soft_skills_score"],
                    recommendation=fallback_evaluation["recommendation"],
                    message_suffix=" (Fallback processing)"
                )
            except Exception as fe:
                self.log(f"Fallback also failed: {fe}")
                return self.create_error_result(f"Processing failed: {str(e)}", "Processing Error")


# --- 7. Legacy/Wrapper Functions ---

def create_fallback_data(data_type="evaluation", **kwargs):
    return DataProcessor().create_fallback_data(data_type, **kwargs)

def parse_json_response(response_text, function_name="Unknown"):
    return DataProcessor().parse_json_response(response_text, function_name)

def manage_excel_data(results_data=None, role=None, threshold=None, action="save"):
    return DataProcessor().manage_data(action=action, results_data=results_data, role=role, threshold=threshold)

def manage_historical_data(role=None, all_results=None, action="load", top_n=5):
    return DataProcessor().manage_data(action=action, role=role, all_results=all_results, top_n=top_n)

def load_job_data(filename="job_descriptions/all_jobs.json"):
    return DataProcessor().load_job_data(filename)

def get_job_info(role=None, title=None, action="load"):
    return DataProcessor().get_job_info(role=role, title=title, action=action)

def extract_file_content(file_path):
    return DataProcessor().extract_file_content(file_path)

def compute_similarity(resume_text, job_description, model_name=None):
    return DataProcessor(model_name).compute_similarity(resume_text, job_description)

def extract_ner(text, model_name=None, role=None):
    return DataProcessor(model_name).extract_ner(text, role)

def create_error_result(error_message, error_type="General"):
    return DataProcessor().create_error_result(error_message, error_type)

def process_uploaded_resume(file_path, role, threshold, model_name=None):
    return DataProcessor(model_name).process_uploaded_resume(file_path, role, threshold)