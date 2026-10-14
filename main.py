import html

import streamlit as st

from models.kMeans import initialize_model
from models.input import analyze


@st.cache_resource
def get_model():
    return initialize_model()


model = get_model()
Symptoms = model.symptom_columns


def run_task(selected_symptoms):
    return analyze(selected_symptoms, model)

st.set_page_config(layout="wide")

st.title("Disease Classification")
st.markdown(
    """
    <style>
    [data-testid="stColumn"] {
        background: #FFF9F2 !important;
        padding: 1.5rem;
        border: 1px solid rgba(240, 216, 200, 0.9);
        border-radius: 0.9rem;
        box-shadow: 0 5px 18px rgba(74, 3, 3, 0.14);
    }
    [data-testid="stColumn"] h2,
    [data-testid="stColumn"] h3 {
        color: #4A0303;
        letter-spacing: 0;
    }
    [data-testid="stMultiSelect"] > label {
        color: #5F4A43;
        font-weight: 600;
    }
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background: #FFFFFF;
        border: 1px solid #D8B7A5;
        border-radius: 0.55rem;
        min-height: 3rem;
        transition: border-color 160ms ease, box-shadow 160ms ease;
    }
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
        border-color: #880808;
        box-shadow: 0 0 0 3px rgba(136, 8, 8, 0.2), 0 0 0 6px rgba(136, 8, 8, 0.08);
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        display: none;
    }
    .selected-symptoms {
        margin: 0.9rem 0 0.2rem;
        padding: 0.8rem 0.9rem;
        background: #FFFDF9;
        border: 1px solid #F0D8C8;
        border-radius: 0.55rem;
    }
    .selected-symptoms-label {
        margin: 0 0 0.55rem;
        color: #5F4A43;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .selected-symptom {
        display: inline-block;
        margin: 0.2rem 0.35rem 0.2rem 0;
        padding: 0.35rem 0.6rem;
        background: #F7E4D7;
        border: 1px solid #E8C8B5;
        border-radius: 999px;
        color: #4A0303;
        font-size: 0.88rem;
    }
    [data-testid="stButton"] > button {
        width: 100%;
        margin-top: 0.8rem;
        background: #880808;
        border: 1px solid #880808;
        border-radius: 0.55rem;
        color: #FFFFFF;
        font-weight: 700;
        min-height: 2.8rem;
        transition: background 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    }
    [data-testid="stButton"] > button:hover {
        background: #4A0303;
        border-color: #4A0303;
        box-shadow: 0 5px 12px rgba(74, 3, 3, 0.2);
        transform: translateY(-1px);
    }
    .result-card {
        background: #FFFFFF;
        border: 1px solid #F0D8C8;
        border-left: 0.35rem solid #880808;
        border-radius: 0.65rem;
        padding: 0.9rem 1rem;
        margin: 0.7rem 0;
        box-shadow: 0 3px 10px rgba(74, 3, 3, 0.08);
    }
    .result-card h4 {
        color: #4A0303;
        margin: 0 0 0.35rem;
        font-size: 1rem;
    }
    .result-card p {
        color: #5F4A43;
        margin: 0.2rem 0;
    }
    .section-card {
        background: #FFFDF9;
        border: 1px solid #F0D8C8;
        border-radius: 0.65rem;
        padding: 0.8rem 1rem;
        margin: 0.8rem 0;
    }
    .classification-card {
        background: #FFFDF9;
        border: 1px solid #F0D8C8;
        border-left: 0.35rem solid #D97706;
        border-radius: 0.65rem;
        padding: 1rem;
        margin: 0.8rem 0;
        box-shadow: 0 3px 10px rgba(74, 3, 3, 0.08);
    }
    .classification-card h3 {
        color: #4A0303;
        margin: 0 0 0.8rem;
    }
    .category-block {
        border-top: 1px solid #F0D8C8;
        padding: 0.7rem 0;
    }
    .category-block h4 {
        color: #880808;
        margin: 0 0 0.35rem;
    }
    .category-block p {
        color: #5F4A43;
        margin: 0.2rem 0 0.2rem 0.75rem;
    }
    .accuracy {
        border-top: 1px solid #F0D8C8;
        color: #4A0303;
        margin: 0.4rem 0 0;
        padding-top: 0.8rem;
    }
    [data-testid="stApp"] {
        background: linear-gradient(to bottom right, #880808 0%, #880808 65%, #4A0303 100%);
    }
    [data-testid="stHeader"] {
        background-color: #880808;
        color: white;    
    }
    [id=":r1:"] {
        color: white;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)
input_column, results_column = st.columns(2)

with input_column:
    st.subheader("Select Symptoms")
    symptoms = st.multiselect(
        "Symptoms",
        Symptoms,
        key="symptoms",
        default=[Symptoms[0]],
    )
    if symptoms:
        selected_symptoms = "".join(
            f'<span class="selected-symptom">{html.escape(symptom)}</span>'
            for symptom in symptoms
        )
        st.markdown(
            f'<div class="selected-symptoms">'
            f'<p class="selected-symptoms-label">Selected symptoms</p>'
            f'{selected_symptoms}</div>',
            unsafe_allow_html=True,
        )
    analyze_pressed = st.button("Analyze", type="primary")

with results_column:
    st.subheader("Results")
    if analyze_pressed:
        summary, matches, validation_accuracy = run_task(symptoms)
        st.markdown('<div class="section-card"><h4>Closest disease matches</h4>', unsafe_allow_html=True)
        for disease, distance in matches:
            st.markdown(
                f'<div class="result-card"><h4>{disease}</h4>'
                f'<p>Relative distance: <strong>{distance:.2f}</strong></p></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="classification-card"><h3>Classification represented</h3>',
            unsafe_allow_html=True,
        )
        category_names = ("By cause", "By duration/course", "By body system")
        for category_name, category_values in zip(category_names, summary):
            st.markdown(
                f'<div class="category-block"><h4>{category_name}</h4>',
                unsafe_allow_html=True,
            )
            for value in sorted(category_values):
                st.markdown(f"<p>{value}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="accuracy"><strong>Model validation accuracy:</strong> '
            f'{validation_accuracy:.1%}</p></div>',
            unsafe_allow_html=True,
        )


