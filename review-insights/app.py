import json

import pandas as pd
import streamlit as st

import urllib.parse

from llm import BUSINESS_CONTEXT, get_recommendation
from shared import aggregate_keywords, sample_batch

MAX_CALLS = 5  # per visitor session

st.set_page_config(
    page_title="Customer Review Insights",
    page_icon=":material/bar_chart:",
    layout="wide",
)

# Only custom CSS: sidebar width and plain, equal-size sidebar links.
st.markdown(
    """<style>
    [data-testid="stSidebar"] {min-width: 240px !important; max-width: 240px !important; width: 240px !important;}
    [data-testid="stSidebar"] a {text-decoration: none; color: inherit; font-size: 1.25rem; font-weight: 600;}
    </style>""",
    unsafe_allow_html=True,
)

DATA_URL = "https://www.kaggle.com/datasets/shitalkat/amazonearphonesreviews/data"


@st.cache_data
def load_pool():
    return pd.read_csv("artifacts/test_pool.csv")


@st.cache_data
def load_metrics():
    with open("artifacts/metrics.json") as f:
        return json.load(f)


def new_batch():
    st.session_state.batch = sample_batch(load_pool())
    st.session_state.analyzed = False
    st.session_state.plan = None

def batch_scores(b):
    y, p = b["label"], b["pred"]
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return float((y == p).mean()), f1


if "batch" not in st.session_state:
    new_batch()
batch = st.session_state.batch

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("[:material/home: Dashboard](#reviews)")
    st.markdown("[:material/chat: Reviews](#reviews)")
    st.markdown("[:material/bar_chart: Analysis](#analysis)")

# ---------- Section 1: Reviews ----------
st.header("Reviews", anchor="reviews")
st.write("Turn customer reviews into clear findings and next steps for your product team.")

st.subheader("Add Business Context (example)")
st.text_area("Business context", BUSINESS_CONTEXT, disabled=True, height=110, label_visibility="collapsed")

st.subheader("Customer reviews (simulated upload)")
st.caption(
    "A simulation of reviews uploaded from an online store such as Amazon: "
    "20 reviews of the JBL T110BT earphones from a held-out test set. "
    f"Source: [Kaggle Amazon Earphones Reviews]({DATA_URL})."
)

st.button("Shuffle reviews", icon=":material/shuffle:", on_click=new_batch)

reviews = batch[["text"]].rename(columns={"text": "Online review"})
reviews.index = range(1, len(reviews) + 1)
st.dataframe(
    reviews,
    height=400,
    width="stretch",
    column_config={"Online review": st.column_config.TextColumn(width="large")},
)

st.write("")

if st.button("Analyze these reviews", icon=":material/play_arrow:", type="primary"):
    st.session_state.analyzed = True

# ---------- Section 2: Analysis ----------
st.header("Analysis", anchor="analysis")
left, right = st.columns(2)

with left:
    st.subheader("Model Results")
    if not st.session_state.get("analyzed"):
        st.info("Press **Analyze these reviews** to see how the model classified them.")
    else:
        acc, f1 = batch_scores(batch)
        m = load_metrics()
        c1, c2 = st.columns(2)
        c1.metric("Accuracy vs. dataset labels", f"{acc:.0%}")
        c2.metric("F1 vs. dataset labels", f"{f1:.2f}")
        st.caption(
            # f"On all {m['n_test']} held-out test reviews: "
            # f"accuracy {m['distilbert']['accuracy']:.1%}, F1 {m['distilbert']['f1']:.2f}. "
            # "Accuracy is the share of reviews classified correctly. "
            # "F1 balances missed and wrongly flagged reviews; closer to 1 is better."
            "Scored on these 20 reviews. Uploaded reviews normally have no known answer;"
            "here the dataset's star ratings (1-2 stars negative, 4-5 stars positive) serve as the answer key."
        )

        with st.expander("See each prediction"):
            table = pd.DataFrame(
                {
                    "Online review": batch["text"],
                    "Model says": batch["pred"].map({0: "Negative", 1: "Positive"}),
                    "Confidence": batch.apply(
                        lambda r: r["prob_pos"] if r["pred"] == 1 else 1 - r["prob_pos"], axis=1
                    ),
                    "Actual": batch["label"].map({0: "Negative", 1: "Positive"}),
                }
            )
            table.index = range(1, len(table) + 1)
            st.dataframe(
                table,
                width="stretch",
                column_config={
                    "Online review": st.column_config.TextColumn(width="large"),
                    "Confidence": st.column_config.NumberColumn(format="percent"),
                },
            )

        st.markdown("**What the model focused on**")
        keywords = aggregate_keywords(batch)
        for name, color in (("negative", "#EF4444"), ("positive", "#22C55E")):
            kws = keywords[name]["keywords"][:10]
            st.caption(f"Words that pushed reviews toward {name} ({keywords[name]['n_reviews']} reviews)")
            if kws:
                df = pd.DataFrame(kws).set_index("word")[["total_shap"]]
                st.bar_chart(df, horizontal=True, color=color, height=280)
        st.caption(
            "Longer bar = the word influenced the model's decision more. "
            "This shows what the model relied on, not proof that a word is the cause."
        )

        with st.expander("About the classification model"):
            st.markdown(
                "**How it was built**\n"
                "1. **Data:** 1610 JBL T110BT product Amazon reviews from [Kaggle]({DATA_URL}). Reviews with 3-star ratings were removed; 1-2 stars labeled as negative, 4-5 stars as positive.\n"
                "2. **Split:** 70% train (1127) | 15% validation (241) | 15% test (242), while keeping the positive/negative class balance in each set.\n"
                "3. **Transfer learning:** base model is DistilBERT and fine-tuned it for 3 epochs.\n"
                "4. **Compared with a simple baseline model for benchmarking:** TF-IDF + Logistic Regression.\n"
                "5. **Tested once** on the 242 test set reviews: DistilBERT 94.6% accuracy vs. Logistic Regression 92.1%.\n"
                "6. **Predictions saved in advance**, so this demo runs the 20 simulated customer reviews above (from test set) fast and free."
            )
        with st.expander("About the word highlights (SHAP technique)"):
            st.markdown(
                "SHAP measures how much each word pushed the model toward its decision.\n\n"
                "**How it works:** SHAP hides different combinations of words in a review, asks the model "
                "to predict again each time, and averages how much each word changed the result.\n\n"
                "**Example:** for *\"Worst sound quality. Returned after a few minutes.\"* the model says "
                "**negative**. When \"Worst\" is hidden, the model becomes much less sure the review is negative. "
                "That drop is why \"Worst\" gets the biggest score. Hiding \"after\" or \"few\" barely changes "
                "the prediction, so those words get scores near zero.\n\n"
                "**In the charts:** each bar adds up a word's scores across the reviews in that group. "
                "A longer bar means the word influenced the model more."
            )

with right:
    st.subheader("Action Plan")
    if not st.session_state.get("analyzed"):
        st.info("Press **Analyze these reviews** first.")
    else:
        calls = st.session_state.get("llm_calls", 0)
        if st.button("Generate action plan", icon=":material/lightbulb:", type="primary", disabled=calls >= MAX_CALLS):
            with st.spinner("Reading the reviews and writing recommendations..."):
                try:
                    st.session_state.plan = get_recommendation(
                        aggregate_keywords(batch), st.secrets["OPENAI_API_KEY"]
                    )
                    st.session_state.llm_calls = calls + 1
                except Exception:
                    st.error("Could not generate recommendations right now. Please try again later.")
        if calls >= MAX_CALLS:
            st.warning("Demo limit reached for this session.")

        plan = st.session_state.get("plan")
        if plan:
            st.markdown(plan)
            st.caption("Written by an AI model (gpt-4o-mini) from the word summary of these 20 reviews.")
            st.download_button("Download in .txt", plan, file_name="action_plan.txt", icon=":material/download:")
            mail = "mailto:?subject=" + urllib.parse.quote("Review insights: action plan") + "&body=" + urllib.parse.quote(plan)
            st.link_button("Send as an Email", mail, icon=":material/mail:")