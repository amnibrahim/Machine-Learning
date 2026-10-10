BUSINESS_CONTEXT = (
    "Client is a product team of the JBL T110BT, an entry-level wired in-ear earphone sold online. The team wants to know what customers praise and complain about in recent reviews, so they can prioritize product improvements, quality control, and customer support actions."
)

SYSTEM_PROMPT = """You are a customer-review analyst. You receive a business context and a JSON summary of a small sample of reviews that a sentiment model classified and explained with SHAP.

JSON fields:
- "sample_size": total reviews in the sample (about 10 per class). All conclusions rest on this small sample.
- "negative" / "positive": reviews grouped by the model's predicted class. Predictions can be wrong.
- "n_reviews": number of reviews in that class.
- "keywords": words that pushed the model toward that class, ranked by "total_shap". Each has:
  - "word": the keyword itself (a single word taken from tokenized text, so a few may be word pieces).
  - "reviews_containing": how many reviews in the class contain it. The sample is small, so counts are small by design: 3 of 10 is a notable share, and 1 is a hint, not noise.
  - "total_shap": summed SHAP score; higher means a stronger push toward the class.
- "example_reviews": high-confidence reviews per class, for context.

How to use the keywords: group related words into themes (e.g. "stopped", "working", "defective" into a reliability theme). 
Judge each word using the example reviews and the other keywords. 
Words in only 1 review may be used when they fit a theme, but say the evidence is thin. 
Keywords are single words, so use the example reviews to see what words like "not" refer to.

Answer in plain text with these sections:
1. Negative reviews: main complaint themes (2 to 4 sentences).
2. Positive reviews: main praise themes (2 to 4 sentences).
3. Recommended actions: 1 to 3 prioritized actions tailored to the business context. For each, cite the supporting keywords or example reviews.

Rules: 
- Base every statement on the provided data; do not invent facts, numbers, or product details; be concise and specific.
- Write for busy business readers: plain everyday language, no technical terms (do not mention SHAP, tokens, or the model), and short sentences. 
- Keep the whole answer under 150 words."""

import json
import tomllib
from pathlib import Path

from openai import OpenAI

def build_user_message(payload):
    return (
        f"Business context:\n{BUSINESS_CONTEXT}\n\n"
        f"Review summary (JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

def get_recommendation(payload, api_key):
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=400,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(payload)},
        ],
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    import pandas as pd
    from shared import aggregate_keywords, sample_batch

    secrets = tomllib.loads(Path(".streamlit/secrets.toml").read_text())
    pool = pd.read_csv("artifacts/test_pool.csv")
    payload = aggregate_keywords(sample_batch(pool, seed=2))
    print(get_recommendation(payload, secrets["OPENAI_API_KEY"]))