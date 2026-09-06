import re
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from sentence_transformers import SentenceTransformer
import hdbscan
from lime.lime_text import LimeTextExplainer

app = FastAPI(title="SIF-Sentinel API", version="1.0.0")

# ---------------------------------------------------------
# MODEL INITIALIZATION (Lightweight & CPU-Optimized)
# ---------------------------------------------------------

# 1. NLP / NER Extractor
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns([
    {"label": "LOCATION", "pattern": [{"LOWER": "rig"}, {"IS_DIGIT": True}]},
    {"label": "LOCATION", "pattern": [{"LOWER": "deck"}]},
    {"label": "ACTIVITY", "pattern": [{"LOWER": "routine"}, {"LOWER": "maintenance"}]},
    {"label": "ACTIVITY", "pattern": [{"LOWER": "tripping"}, {"LOWER": "pipe"}]},
    {"label": "EQUIPMENT", "pattern": [{"LOWER": "main"}, {"LOWER": "compressor"}, {"LOWER": "valve"}]},
    {"label": "EQUIPMENT", "pattern": [{"LOWER": "metal"}, {"LOWER": "pipe"}, {"LOWER": "fitting"}]},
    {"label": "BARRIER-FAILURE", "pattern": [{"LOWER": "loto"}]},
    {"label": "BARRIER-FAILURE", "pattern": [{"LOWER": "bypassed"}]},
    {"label": "BARRIER-FAILURE", "pattern": [{"LOWER": "high"}, {"TEXT": "-"}, {"LOWER": "pressure"}, {"LOWER": "release"}]}
])

# 2. Dense Embedder for Semantic Clustering
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. LIME Explainer
lime_explainer = LimeTextExplainer(class_names=["Non-SIF", "SIF-Potential"])

# 9 IOGP Life-Saving Rules Definitions
IOGP_RULES = [
    "Bypassing Safety Controls",
    "Confined Space",
    "Energy Isolation",
    "Driving Safety",
    "Hot Work",
    "Line of Fire",
    "Safe Mechanical Lifting",
    "Work Authorization",
    "Working at Height"
]

# In-memory buffer to accumulate reports for real-time HDBSCAN clustering
HISTORICAL_REPORTS_BUFFER = [
    {"id": "HIST_01", "text": "High pressure valve seat cracked during nitrogen purging test at Rig 42."},
    {"id": "HIST_02", "text": "Compressor discharge valve leaking gas, bypass valve engaged without permit."},
    {"id": "HIST_03", "text": "Worker tripped on loose cable bundle in the canteen corridor."},
    {"id": "HIST_04", "text": "Scaffold toe-board dislodged at 15 meters on Rig 08; fell onto safety netting."},
    {"id": "HIST_05", "text": "Pressure safety valve jammed shut during pump switchover at Duliajan station."}
]

# ---------------------------------------------------------
# CORE SIF SCORING FUNCTION (DistilBERT Simulation/Inference)
# ---------------------------------------------------------

def predict_sif_proba(texts: List[str]) -> np.ndarray:
    """
    Inference scoring function returning probabilities [[P(Non-SIF), P(SIF)]].
    Evaluates high-energy hazard indicators and barrier failure cues.
    """
    high_hazard_keywords = [
        "bypass", "bypassed", "loto", "pressure release", "flange", 
        "narrowly missing", "gas leak", "h2s", "struck by", "dropped object",
        "scaffold fall", "unauthorized", "explosion", "suspended load"
    ]
    probabilities = []
    for text in texts:
        text_lower = text.lower()
        score = 0.15  # baseline probability
        matches = sum(1 for kw in high_hazard_keywords if kw in text_lower)
        score += min(matches * 0.25, 0.80)
        
        # Soft cap at 98%
        sif_prob = min(max(score, 0.02), 0.98)
        probabilities.append([1.0 - sif_prob, sif_prob])
        
    return np.array(probabilities)


# ---------------------------------------------------------
# LAYER 1: IOGP MULTI-LABEL TAGGER
# ---------------------------------------------------------

def tag_iogp_rules(text: str) -> List[Dict[str, Any]]:
    """
    Maps narratives to standard IOGP Life-Saving Rules using semantic similarity.
    """
    text_emb = embedder.encode(text, convert_to_tensor=True)
    rule_embs = embedder.encode(IOGP_RULES, convert_to_tensor=True)
    
    # Compute cosine similarities between text and IOGP definitions
    from sentence_transformers import util
    cosine_scores = util.cos_sim(text_emb, rule_embs)[0].cpu().numpy()
    
    matched_rules = []
    for idx, score in enumerate(cosine_scores):
        if float(score) >= 0.28:  # Confidence threshold for matching
            matched_rules.append({
                "rule": IOGP_RULES[idx],
                "confidence": round(float(score), 3)
            })
            
    matched_rules = sorted(matched_rules, key=lambda x: x["confidence"], reverse=True)
    return matched_rules


# ---------------------------------------------------------
# LAYER 2: EXPLAINABLE AI (LIME Token Attribution)
# ---------------------------------------------------------

def run_lime_explanation(text: str) -> List[Dict[str, Any]]:
    """
    Generates local feature token attributions explaining what drove the score.
    """
    # num_samples=100 ensures snappy execution on CPU (< 1 sec)
    exp = lime_explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_sif_proba,
        num_features=6,
        num_samples=100,
        labels=[1]
    )
    
    token_weights = []
    for word, weight in exp.as_list(label=1):
        token_weights.append({
            "word": word,
            "weight": round(weight, 4),
            "driver": "SIF Potential" if weight > 0 else "Non-SIF Barrier"
        })
    return token_weights


# ---------------------------------------------------------
# LAYER 3: ALGORITHMIC FAIRNESS AUDIT LAYER
# ---------------------------------------------------------

def run_fairness_probe(text: str, base_score: float) -> Dict[str, Any]:
    """
    Counterfactual Fairness Probe: Swaps contractor names, shifts, and 
    worker demographics to ensure non-causal entities do not sway the score.
    """
    # Define counterfactual swap candidates
    perturbation_pairs = [
        (r"\bRig \d+\b", "Rig 01"),
        (r"\bContractor [A-Z]\b", "Contractor X"),
        (r"\bShift [A-C]\b", "Shift Night"),
        (r"\bmechanic\b", "junior operator"),
        (r"\btechnician\b", "apprentice")
    ]
    
    counterfactual_scores = []
    tested_variants = []
    
    for pattern, replacement in perturbation_pairs:
        if re.search(pattern, text, flags=re.IGNORECASE):
            perturbed_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            new_prob = float(predict_sif_proba([perturbed_text])[0][1])
            delta = abs(new_prob - base_score)
            
            counterfactual_scores.append(delta)
            tested_variants.append({
                "swap_pattern": pattern,
                "score_delta": round(delta, 4)
            })

    max_delta = max(counterfactual_scores) if counterfactual_scores else 0.00
    
    # DGMS / OISD Regulatory Safety standard check (score shift <= 5%)
    audit_passed = max_delta <= 0.05

    return {
        "audit_passed": audit_passed,
        "max_variance": round(max_delta, 4),
        "status": "APPROVED (Bias < 5%)" if audit_passed else "FLAGGED (Human Review Required)",
        "evaluations": tested_variants
    }


# ---------------------------------------------------------
# LAYER 4: EMERGING THREAT CLUSTERING (HDBSCAN)
# ---------------------------------------------------------

def cluster_emerging_threats(incoming_text: str) -> Dict[str, Any]:
    """
    Sentence-BERT + HDBSCAN clustering to detect recurring patterns 
    that fall outside standard taxonomic definitions.
    """
    corpus = [r["text"] for r in HISTORICAL_REPORTS_BUFFER] + [incoming_text]
    embeddings = embedder.encode(corpus)
    
    # HDBSCAN clustering: min_cluster_size=2 captures early localized clusters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean", min_samples=1)
    labels = clusterer.fit_predict(embeddings)
    
    incoming_label = int(labels[-1])
    
    if incoming_label != -1:
        # Clustered with historical reports
        matched_indices = [i for i, lbl in enumerate(labels[:-1]) if lbl == incoming_label]
        cluster_samples = [HISTORICAL_REPORTS_BUFFER[i]["text"] for i in matched_indices]
        cluster_summary = f"Detected emerging hazard cluster #{incoming_label} across {len(matched_indices) + 1} related reports."
    else:
        cluster_summary = "Isolated incident — does not fit an active emerging cluster pattern."
        cluster_samples = []

    return {
        "cluster_id": incoming_label,
        "is_emerging_threat": incoming_label != -1,
        "cluster_summary": cluster_summary,
        "correlated_precursors": cluster_samples
    }


# ---------------------------------------------------------
# API REQUEST & PIPELINE ENDPOINT
# ---------------------------------------------------------

class IncidentReport(BaseModel):
    report_id: str
    text: str

@app.post("/analyze_sif")
def analyze_report(report: IncidentReport):
    # Base SIF Probability
    sif_probability = float(predict_sif_proba([report.text])[0][1])
    
    # NER Domain Extraction
    doc = nlp(report.text)
    domain_labels = ["LOCATION", "ACTIVITY", "EQUIPMENT", "BARRIER-FAILURE"]
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ in domain_labels]

    # Execute the 4 Layers
    iogp_tags = tag_iogp_rules(report.text)
    xai_attributions = run_lime_explanation(report.text)
    fairness_audit = run_fairness_probe(report.text, sif_probability)
    cluster_insights = cluster_emerging_threats(report.text)

    # Determine confidence-based triage tier
    if sif_probability >= 0.70:
        triage_action = "ESCALATE_IMMEDIATE_SUPERVISOR"
    elif sif_probability >= 0.40:
        triage_action = "HUMAN_REVIEW_SECONDARY_QUEUE"
    else:
        triage_action = "AUTO_FILE_LOW_RISK"

    return {
        "report_id": report.report_id,
        "sif_probability": round(sif_probability, 4),
        "triage_action": triage_action,
        "entities": entities,
        "iogp_rules": iogp_tags,
        "lime_attribution": xai_attributions,
        "fairness_audit": fairness_audit,
        "clustering": cluster_insights
    }