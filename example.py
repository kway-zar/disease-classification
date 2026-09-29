import streamlit as st
import streamlit_shadcn_ui as ui
import st_tailwind as tw

from disease_classification import disease_classification

# Add some test code to play with the component while it's in development.
# During development, we can run this just as we would any other Streamlit
# app: `$ streamlit run disease_classification/example.py`

columns = st.columns(2)
with columns[0]:
    ui.card(
        title="Create project",
        content="Start from a clean V2 component surface.",
        description="Standard size",
    )
with columns[1]:
    ui.card(
        title="Activity",
        content="Three components were verified today.",
        description="Compact size",
        size="sm",
    )
