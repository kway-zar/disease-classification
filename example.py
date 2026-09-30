from pandas.core.dtypes.common import classes
import streamlit as st
import streamlit_shadcn_ui as ui
import st_tailwind as tw

from disease_classification import disease_classification

# Add some test code to play with the component while it's in development.
# During development, we can run this just as we would any other Streamlit
# app: `$ streamlit run disease_classification/example.py`
st.set_page_config(layout="wide")




tw.initialize_tailwind()
tw.write("Grid Container", classes="text-blue-500 pb-4")

with tw.container(classes="grid grid-cols-2 gap-x-10 ring-0", height=900):
    with tw.container(classes="bg-white rounded-xl shadow shadow-xl", height="stretch"):
        tw.markdown(
                '<p style="font-size: 3rem; font-weight:semibold;">Enter Disease</p>',
            unsafe_allow_html=True,
            classes="text-black mt-10 ml-10",
        )
        languages = tw.multiselect(
            "",
            ["Python", "Java", "C++", "Rust", "Go"],
            classes="mx-10",
            key=1,
            default=["Python"],  # Optional: sets default selections
        )

        tw.markdown(
            '<p style="font-size: 3rem; font-weight:semibold">Select Symptoms</p>',
            unsafe_allow_html=True,
            classes="text-black mt-10 ml-10",
        )
        symptoms = tw.multiselect(
            "",
            ["Python", "Java", "C++", "Rust", "Go"],
            classes="mx-10",
            key=2,
            default=["Python"],  # Optional: sets default selections
        )
    with tw.container(classes="bg-white rounded-xl shadow shadow-xl", height="stretch"):
        tw.markdown(
                '<p style="font-size: 3rem; font-weight:semibold;">Results</p>',
            unsafe_allow_html=True,
            classes="text-black mt-10 ml-10",
        )

