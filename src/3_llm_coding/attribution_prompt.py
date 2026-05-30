"""
Attribution coding prompt and structured output schema.
Adapted from Bettman & Weitzel (1983) attribution framework for earnings-call discourse.
"""

SYSTEM_PROMPT = """You are an expert analyst trained in corporate disclosure, behavioral accounting, and causal attribution theory. Your task is to read a single analyst question and the executive's response, and extract the manager's attribution structure following the Bettman-Weitzel framework.

For the executive answer, you must extract FIVE fields and output them as a JSON object. ALL FIVE FIELDS ARE REQUIRED — including key_phrase. Do not omit any field.

(1) outcome_valence — what the manager is discussing:
   - "positive": favorable outcome (growth, beat, margin expansion, win, success)
   - "negative": unfavorable outcome (decline, miss, margin compression, loss, setback)
   - "neutral": factual/forward-looking with no clear valence
   - "mixed": both positive and negative outcomes discussed

(2) attribution_target — the cause the manager assigns to the outcome:
   - "internal_action": management decisions, strategy execution, operational choices, our team
   - "internal_capability": firm-level strengths, competitive advantages, products, technology
   - "external_structural": macro, industry dynamics, regulation, geopolitics, supply chain, FX, rates
   - "external_agentic": specific external actors (competitors, customers, suppliers, regulators)
   - "mixed_attribution": multiple targets given roughly equal weight
   - "no_attribution": descriptive only, no causal claim made

(3) attribution_certainty — how committed is the manager to this causal claim:
   - "high": direct causal language ("we drove", "this caused", "because of")
   - "medium": hedged but clear ("contributed to", "factor in")
   - "low": vague/defensive ("environment", "headwinds", "challenges", passive voice)

(4) responsiveness — does the answer address the question:
   - "direct": engages with the specific question
   - "partial": addresses some aspects, deflects others
   - "deflective": pivots to unrelated topic, evades

(5) key_phrase — a verbatim quote (5-15 words, COPIED EXACTLY from the executive's answer) that best supports your attribution classification. This field is MANDATORY.

Example output format:
{"outcome_valence": "negative", "attribution_target": "external_structural", "attribution_certainty": "medium", "responsiveness": "direct", "key_phrase": "supply chain pressures continued to weigh on our margins this quarter"}

Output ONLY the JSON object. No reasoning, no explanation, no markdown code fences."""

USER_PROMPT_TEMPLATE = """ANALYST QUESTION (asked by {q_speaker}):
{q_text}

EXECUTIVE ANSWER (by {a_speaker}):
{a_text}

Extract the attribution structure as JSON."""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome_valence": {
            "type": "string",
            "enum": ["positive", "negative", "neutral", "mixed"]
        },
        "attribution_target": {
            "type": "string",
            "enum": [
                "internal_action", "internal_capability",
                "external_structural", "external_agentic",
                "mixed_attribution", "no_attribution"
            ]
        },
        "attribution_certainty": {
            "type": "string",
            "enum": ["high", "medium", "low"]
        },
        "responsiveness": {
            "type": "string",
            "enum": ["direct", "partial", "deflective"]
        },
        "key_phrase": {
            "type": "string",
            "description": "Short verbatim phrase (≤15 words) supporting the attribution classification"
        }
    },
    "required": ["outcome_valence", "attribution_target",
                 "attribution_certainty", "responsiveness", "key_phrase"]
}


# Self-Serving Attribution score, derived post-hoc from extracted fields
def compute_ssa(row):
    """
    SSA = +1 if manager takes credit for positive outcome (internal × positive)
    SSA = -1 if manager externalizes negative outcome (external × negative)
    SSA = 0 otherwise
    """
    v = row['outcome_valence']
    t = row['attribution_target']
    if v == 'positive' and t in ('internal_action', 'internal_capability'):
        return 1
    if v == 'negative' and t in ('external_structural', 'external_agentic'):
        return -1
    return 0