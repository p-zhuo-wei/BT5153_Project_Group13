"""Text cleaning, PII masking, category inference, and shared lookup tables.

Cell-equivalents from the notebook: rule tables (cell 6) + Step 1A helpers (cell 8).
"""

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import spacy
except Exception:
    spacy = None


# Shared rule tables, category maps, and skill patterns used throughout the notebook.

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it", "of", "on",
    "or", "that", "the", "to", "with", "will", "this", "your", "you", "our", "we", "their", "they", "into",
    "about", "than", "then", "them", "who", "what", "when", "where", "while", "can", "using", "use", "used",
    "within", "across", "over", "under", "per", "day", "month", "year", "full", "time", "part", "job", "work",
    "resume", "curriculum", "vitae", "experience", "skills", "skill", "description", "requirements", "benefits",
}

RESUME_CATEGORY_MAP = {
    "ACCOUNTANT": "accounting_finance",
    "ADVOCATE": "legal_consulting",
    "AGRICULTURE": "operations_engineering",
    "APPAREL": "creative_design_media",
    "ARTS": "creative_design_media",
    "AUTOMOBILE": "operations_engineering",
    "AVIATION": "operations_engineering",
    "BANKING": "accounting_finance",
    "BPO": "hr_customer_support",
    "BUSINESS-DEVELOPMENT": "sales_marketing",
    "CHEF": "hospitality_wellness",
    "CONSTRUCTION": "operations_engineering",
    "CONSULTANT": "legal_consulting",
    "DESIGNER": "creative_design_media",
    "DIGITAL-MEDIA": "sales_marketing",
    "ENGINEERING": "operations_engineering",
    "FINANCE": "accounting_finance",
    "FITNESS": "hospitality_wellness",
    "HEALTHCARE": "healthcare",
    "HR": "hr_customer_support",
    "INFORMATION-TECHNOLOGY": "technology_data",
    "PUBLIC-RELATIONS": "sales_marketing",
    "SALES": "sales_marketing",
    "TEACHER": "education_training",
}

JOB_CATEGORY_RULES = {
    "technology_data": [
        r"software", r"developer", r"engineer", r"devops", r"database", r"network", r"systems?", r"it\b",
        r"information technology", r"web", r"python", r"java", r"javascript", r"ios", r"php", r"django",
        r"machine learning", r"technical", r"data scientist", r"data engineer", r"administrator",
    ],
    "sales_marketing": [
        r"marketing", r"sales", r"account executive", r"account manager", r"business development", r"advertising",
        r"brand", r"growth", r"seo", r"sem", r"public relations", r"social media", r"customer acquisition",
    ],
    "accounting_finance": [
        r"account(?:ant|ing)", r"finance", r"financial", r"auditor", r"bank", r"bookkeep", r"payroll",
        r"treasury", r"tax", r"controller",
    ],
    "hr_customer_support": [
        r"human resources", r"hr\b", r"recruit", r"talent", r"administrative", r"admin\b", r"office manager",
        r"customer service", r"support", r"success", r"receptionist", r"operations support",
    ],
    "education_training": [
        r"teacher", r"education", r"instructor", r"tutor", r"training", r"professor", r"school", r"curriculum",
        r"e-learning", r"academic",
    ],
    "healthcare": [
        r"health", r"medical", r"clinical", r"hospital", r"nurs", r"patient", r"dental", r"therap",
        r"physician", r"care provider",
    ],
    "creative_design_media": [
        r"design", r"designer", r"creative", r"art", r"fashion", r"apparel", r"content", r"editor",
        r"writer", r"graphic", r"ux", r"ui", r"video", r"media",
    ],
    "operations_engineering": [
        r"construction", r"operations", r"project manager", r"manufacturing", r"mechanic", r"commissioning",
        r"logistics", r"procurement", r"automotive", r"aviation", r"agri", r"warehouse", r"quality",
        r"civil", r"electrical", r"mechanical", r"production",
    ],
    "hospitality_wellness": [
        r"chef", r"cook", r"restaurant", r"hospitality", r"hotel", r"food", r"fitness", r"trainer", r"spa",
        r"wellness",
    ],
    "legal_consulting": [
        r"legal", r"advocate", r"attorney", r"consultant", r"compliance", r"counsel", r"advisory", r"policy",
    ],
}

RELATED_CATEGORIES = {
    "accounting_finance": {"sales_marketing", "legal_consulting", "hr_customer_support"},
    "creative_design_media": {"sales_marketing", "education_training", "technology_data"},
    "education_training": {"hr_customer_support", "creative_design_media", "healthcare"},
    "healthcare": {"education_training", "hospitality_wellness", "hr_customer_support"},
    "hospitality_wellness": {"sales_marketing", "healthcare", "hr_customer_support"},
    "hr_customer_support": {"sales_marketing", "education_training", "accounting_finance"},
    "legal_consulting": {"accounting_finance", "sales_marketing", "hr_customer_support"},
    "operations_engineering": {"technology_data", "sales_marketing", "accounting_finance"},
    "sales_marketing": {"hr_customer_support", "creative_design_media", "accounting_finance"},
    "technology_data": {"operations_engineering", "creative_design_media", "sales_marketing"},
}

SKILL_PATTERNS = {
    "python": r"\bpython\b",
    "sql": r"\bsql\b|structured query language",
    "java": r"\bjava\b",
    "javascript": r"\bjavascript\b|\bjs\b",
    "typescript": r"\btypescript\b|\bts\b",
    "react": r"\breact\b",
    "nodejs": r"node\s?js|nodejs",
    "django": r"\bdjango\b",
    "php": r"\bphp\b",
    "aws": r"\baws\b|amazon web services",
    "azure": r"\bazure\b",
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "linux": r"\blinux\b",
    "excel": r"\bexcel\b|spreadsheets?",
    "powerbi": r"power\s?bi",
    "tableau": r"\btableau\b",
    "tensorflow": r"\btensorflow\b",
    "pytorch": r"\bpytorch\b",
    "machine_learning": r"machine learning|ml\b",
    "project_management": r"project management|project manager",
    "agile": r"\bagile\b|\bscrum\b",
    "sales": r"\bsales\b|lead generation|pipeline management",
    "marketing": r"marketing|seo|sem|campaign|brand management",
    "customer_service": r"customer service|client service|call center|help desk",
    "recruiting": r"recruiting|talent acquisition|sourcing|onboarding",
    "payroll": r"payroll|accounts payable|accounts receivable",
    "auditing": r"audit|auditing|internal control",
    "accounting": r"accounting|bookkeeping|financial reporting|budgeting",
    "consulting": r"consulting|advisory|stakeholder management",
    "legal_research": r"legal research|litigation|compliance|contract drafting",
    "teaching": r"teaching|curriculum|lesson planning|classroom",
    "patient_care": r"patient care|clinical|medical records|nursing",
    "fitness": r"fitness|personal training|exercise programming|nutrition",
    "cooking": r"chef|culinary|food preparation|kitchen management",
    "autocad": r"autocad|cad\b",
    "manufacturing": r"manufacturing|quality control|lean|six sigma",
    "logistics": r"logistics|supply chain|warehouse|inventory",
    "graphic_design": r"graphic design|photoshop|illustrator|indesign",
    "content_creation": r"content creation|copywriting|editorial|social media",
}

CATEGORY_DISPLAY = {
    "accounting_finance": "Accounting / Finance",
    "creative_design_media": "Creative / Design / Media",
    "education_training": "Education / Training",
    "healthcare": "Healthcare",
    "hospitality_wellness": "Hospitality / Wellness",
    "hr_customer_support": "HR / Customer Support",
    "legal_consulting": "Legal / Consulting",
    "operations_engineering": "Operations / Engineering",
    "sales_marketing": "Sales / Marketing",
    "technology_data": "Technology / Data",
}

# Core preprocessing helpers for cleaning text and inferring normalized categories.

def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = text.replace("&amp;", " and ")
    text = re.sub(r"[_\-]{2,}", " ", text)
    text = re.sub(r"\bcompany name\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_spacy_model(enabled: bool = True):
    if not enabled or spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
    except Exception:
        return None


def mask_pii(text: str, nlp=None) -> str:
    masked = text
    pii_patterns = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"(?:(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3}[\s.-]?\d{4,6}(?!\w))",
        "URL": r"\b(?:https?://|www\.)\S+\b",
        "ZIP": r"\b\d{5}(?:-\d{4})?\b",
    }
    for label, pattern in pii_patterns.items():
        masked = re.sub(pattern, f" [{label}] ", masked, flags=re.IGNORECASE)

    if nlp is not None and masked:
        doc = nlp(masked)
        replacements = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "GPE", "LOC", "ORG"}:
                replacements.append((ent.start_char, ent.end_char, f" [{ent.label_}] "))
        if replacements:
            pieces = []
            cursor = 0
            for start, end, replacement in sorted(replacements, key=lambda item: item[0]):
                if start < cursor:
                    continue
                pieces.append(masked[cursor:start])
                pieces.append(replacement)
                cursor = end
            pieces.append(masked[cursor:])
            masked = "".join(pieces)

    masked = re.sub(r"\s+", " ", masked)
    return masked.strip()


def extract_skills(text: str) -> list[str]:
    lowered = text.lower()
    matches = [skill for skill, pattern in SKILL_PATTERNS.items() if re.search(pattern, lowered, flags=re.IGNORECASE)]
    return sorted(set(matches))


def token_set(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z]{3,}", text.lower()))
    return {token for token in tokens if token not in STOPWORDS}


def normalize_resume_category(category: str) -> str:
    return RESUME_CATEGORY_MAP.get(str(category).strip().upper(), "other")


def infer_job_category(row: pd.Series) -> str:
    fields = [row.get("title", ""), row.get("department", ""), row.get("industry", ""), row.get("function", "")]
    combined = " ".join(str(value) for value in fields if pd.notna(value)).lower()
    scores = {category: 0 for category in JOB_CATEGORY_RULES}

    for category, patterns in JOB_CATEGORY_RULES.items():
        for pattern in patterns:
            if re.search(pattern, combined, flags=re.IGNORECASE):
                scores[category] += 2

    function_value = str(row.get("function", "")).strip().lower()
    function_map = {
        "information technology": "technology_data",
        "engineering": "operations_engineering",
        "sales": "sales_marketing",
        "marketing": "sales_marketing",
        "customer service": "hr_customer_support",
        "administrative": "hr_customer_support",
        "human resources": "hr_customer_support",
        "accounting/auditing": "accounting_finance",
        "finance": "accounting_finance",
        "education": "education_training",
        "health care provider": "healthcare",
        "design": "creative_design_media",
        "art/creative": "creative_design_media",
        "writing/editing": "creative_design_media",
        "project management": "operations_engineering",
        "consulting": "legal_consulting",
    }
    if function_value in function_map:
        scores[function_map[function_value]] += 3

    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else "other"


def overlap_score(left_skills: set[str], right_skills: set[str], left_tokens: set[str], right_tokens: set[str]) -> float:
    skill_overlap = len(left_skills & right_skills)
    skill_union = max(len(left_skills | right_skills), 1)
    token_overlap = len(left_tokens & right_tokens)
    token_union = max(len(left_tokens | right_tokens), 1)
    return 0.7 * (skill_overlap / skill_union) + 0.3 * (token_overlap / token_union)


# General-purpose text helpers (moved from stage_2_llm_rerank for reuse).
import math


def normalize_text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def truncate_text(text, max_chars: int) -> str:
    cleaned = normalize_text(text)
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."
