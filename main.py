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
        background-color: #FFF9F2;
        padding: 1.25rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    }
    [data-testid="stApp"] {
        background-color: #880808;
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
    analyze_pressed = st.button("Analyze", type="primary")

with results_column:
    st.subheader("Results")
    if analyze_pressed:
        summary, matches, validation_accuracy = run_task(symptoms)
        st.write("Closest disease matches:")
        for disease, distance in matches:
            st.write(f"- {disease} (relative distance: {distance:.2f})")
        st.write("Classification represented:")
        category_names = ("By cause", "By duration/course", "By body system")
        for category_name, category_values in zip(category_names, summary):
            st.write(f"{category_name}:")
            for value in sorted(category_values):
                st.write(f"- {value}")
        st.write(f"Model validation accuracy: {validation_accuracy:.1%}")


