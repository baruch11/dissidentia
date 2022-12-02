import sys
import os
import warnings
import pandas as pd
import streamlit as st
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

import warnings

from dissidentia.domain.model_wrapper import DissidentModelWrapper
from dissidentia.infrastructure.grand_debat import get_rootdir

warnings.simplefilter(action="ignore", category=FutureWarning)


class Application:
    """This class define a process for adding a page on the application"""

    def __init__(self) -> None:
        self.pages = []

    def add_page(self, title, name_page):
        """add page on the application"""

        self.pages.append({"title": title, "name_page": name_page})

    def run(self):
        """define and run the application"""
        st.sidebar.markdown("### Menu")
        page = st.sidebar.selectbox(
            label="Selectionner une page",
            options=self.pages,
            format_func=lambda x: x["title"],
        )
        st.sidebar.markdown("___")
        page["name_page"]


def results_page(file_model="BertTypeClassifier"):
    """The main page of the application"""
    st.set_page_config(layout="wide")
    left_col, right_col = st.columns(2)
    left_col.image(os.path.join(get_rootdir(), "data/images/dissidentIA.png"))

    right_col.markdown("# DissidentIA")
    right_col.markdown("### Détection de dissidents politiques dans les réponses au Grand Débat")
    right_col.markdown("**Créée par Charles, Amir et Moindzé**")

    st.markdown("---")

    #st.markdown("### Résultats")

    with st.form(key="my_form"):

        sample = "C'est vraiment nul ! Je suggère de simplifier tout ça."
        message_text = st.text_area(
            "Entrer la reponse à évaluer",
            sample,
            height=100,
            help=" Une reponse issue du dataset du grand débat "
        )

        submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            st.success("✅ Done!")

    if message_text != "":

        model = DissidentModelWrapper.load(file_model)

        result = model.sentensize_and_predict(message_text)

        df = pd.DataFrame(columns=["phrases", "predictions"])

        for j in range(len(result)):
            label = result[j][1]
            text = result[j][0]
            new_row = {"phrases": text, "predictions": label}
            df = df.append(new_row, ignore_index=True)

        st.markdown("##### Prédictions")
        st.table(df)

        if st.button("Explication des prédictions"):
            with st.spinner("Generating explanations"):
                explainer = LimeTextExplainer(
                    class_names=pd.DataFrame([False, True]))
                for j in range(len(result)):
                    text = result[j][0]
                    exp = explainer.explain_instance(
                        text, model.model.predict_proba, num_features=6)
                    components.html(exp.as_html(), height=320)


if __name__ == "__main__":
    app = Application()
    page = results_page(sys.argv[1])
    app.add_page("Resultats", page)
    app.run()
