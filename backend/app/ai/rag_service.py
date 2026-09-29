"""Retrieval-Augmented Generation (RAG) knowledge service for MediSense AI.

Retrieves verified, educational medical literature, clinical guidelines,
and terminology explanations to ground LLM responses.
Does NOT fabricate X-ray findings or diagnostic certainty.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TRUSTED_KNOWLEDGE_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "ref_emergency_chest_pain",
        "topic": "Chest Pain and Acute Coronary Warning Signs",
        "source": "WHO Clinical Guidelines / Emergency Medicine Protocol",
        "content": (
            "Chest pain, especially when accompanied by shortness of breath, pain radiating to the jaw, "
            "neck, or left arm, sweating, or lightheadedness, is a medical emergency requiring immediate "
            "evaluation at an emergency facility. Immediate medical care should not be delayed for home remedies."
        ),
    },
    {
        "id": "ref_respiratory_cough_fever",
        "topic": "Acute Respiratory Symptoms and Persistent Cough",
        "source": "CDC Respiratory Infection Patient Guidance",
        "content": (
            "A cough lasting more than 3-5 days accompanied by fever, productive sputum, or chest tightness "
            "warrants medical consultation with a general physician or pulmonologist to rule out bacterial "
            "infection, bronchitis, or pneumonia. Monitoring hydration, fever response, and oxygenation is standard practice."
        ),
    },
    {
        "id": "ref_fasting_blood_glucose",
        "topic": "Fasting Blood Glucose and Glycemic Ranges",
        "source": "American Diabetes Association (ADA) Clinical Standards",
        "content": (
            "Normal fasting plasma glucose levels are typically between 70 and 99 mg/dL. "
            "Fasting levels between 100 and 125 mg/dL indicate impaired fasting glucose (prediabetes), "
            "while 126 mg/dL or higher on two separate occasions warrants clinical evaluation for diabetes mellitus. "
            "Diet, physical activity, and repeat testing are core clinical next steps."
        ),
    },
    {
        "id": "ref_hemoglobin_anemia",
        "topic": "Hemoglobin Levels and Anemia Assessment",
        "source": "WHO Reference Ranges for Hemoglobin Concentrations",
        "content": (
            "Normal hemoglobin ranges are generally 13.8 to 17.2 g/dL for adult men and 12.1 to 15.1 g/dL "
            "for non-pregnant adult women. Levels below these ranges suggest anemia, which can cause fatigue, "
            "dizziness, and pallor. A physician will commonly investigate nutritional causes (iron, B12), blood loss, or systemic disease."
        ),
    },
    {
        "id": "ref_creatinine_kidney",
        "topic": "Serum Creatinine and Renal Function",
        "source": "National Kidney Foundation (NKF) Clinical Guidelines",
        "content": (
            "Serum creatinine is a waste product of muscle breakdown filtered by the kidneys. "
            "Normal reference ranges are approximately 0.7 to 1.3 mg/dL for men and 0.6 to 1.1 mg/dL for women. "
            "Elevated creatinine may reflect dehydration, acute kidney stress, or chronic renal impairment and requires medical follow-up."
        ),
    },
    {
        "id": "ref_doctor_consultation_prep",
        "topic": "Questions to Ask Your Doctor During a Consultation",
        "source": "Agency for Healthcare Research and Quality (AHRQ)",
        "content": (
            "When consulting a physician regarding new symptoms or abnormal lab tests, recommended questions include: "
            "1. What could be causing these symptoms? "
            "2. Do I need any follow-up tests or repeat lab work? "
            "3. Are there lifestyle modifications that could help? "
            "4. What red-flag warning signs should prompt immediate emergency care?"
        ),
    },
    {
        "id": "ref_xray_explainability_disclaimer",
        "topic": "Artificial Intelligence in Chest Radiography Interpretation",
        "source": "Radiological Society of North America (RSNA) AI Education",
        "content": (
            "AI-based computer vision models for chest X-rays provide assistive decision support by highlighting "
            "suspicious radiographic patterns such as opacity or consolidation. Automated predictions are not "
            "definitive proof of disease and must always be correlated with patient symptoms and verified by a radiologist."
        ),
    },
    {
        "id": "ref_mild_symptom_monitoring",
        "topic": "Home Monitoring and Self-Limiting Illnesses",
        "source": "NHS Guidelines on Self-Care and Monitoring",
        "content": (
            "Mild symptoms such as slight nasal congestion, mild throat tickle, or minor fatigue without fever or chest pain "
            "frequently resolve with adequate rest, hydration, and observation. Over-the-counter comfort items from local pharmacies "
            "provide symptom relief but do not replace medical evaluation if symptoms worsen or fail to improve after 5 to 7 days."
        ),
    },
]


class RAGService:
    """In-memory TF-IDF vector retrieval over verified medical knowledge sources."""

    def __init__(self) -> None:
        self.documents = TRUSTED_KNOWLEDGE_DOCUMENTS
        self.corpus = [f"{d['topic']}. {d['content']}" for d in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        self.knowledge_source_ids = [d["id"] for d in self.documents]

    def is_configured(self) -> bool:
        return True

    def retrieve(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """Retrieve the most relevant trusted medical knowledge chunks for a query."""
        if not query or not query.strip():
            return {
                "status": "empty",
                "chunks": [],
                "knowledge_source_ids": [],
                "query": query,
            }

        query_vec = self.vectorizer.transform([query.strip()])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Get sorted indices
        ranked_indices = similarities.argsort()[::-1]

        results = []
        source_ids = []
        for idx in ranked_indices[:top_k]:
            score = float(similarities[idx])
            if score > 0.05:  # Relevance threshold
                doc = self.documents[idx]
                results.append(
                    {
                        "id": doc["id"],
                        "topic": doc["topic"],
                        "content": doc["content"],
                        "source": doc["source"],
                        "similarity_score": round(score, 3),
                    }
                )
                source_ids.append(doc["id"])

        return {
            "status": "ok" if results else "no_match",
            "chunks": results,
            "knowledge_source_ids": source_ids,
            "query": query,
            "top_k": top_k,
            "message": (
                f"Retrieved {len(results)} relevant medical knowledge citation(s)."
                if results
                else "No high-confidence trusted knowledge matches found for this query."
            ),
        }
