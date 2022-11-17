
import sys
import warnings
import pandas as pd
import streamlit as st

import warnings

from dissidentia.domain.model_wrapper import DissidentModelWrapper

warnings.simplefilter(action="ignore", category=FutureWarning)

class Application:
    """ This class define a process for adding a page on the application"""
    def __init__(self) -> None:
        self.pages = []
    
    def add_page(self, title, name_page):
        """ add page on the application """

        self.pages.append({"title": title, "name_page" : name_page})

    def run(self):
        """ define and run the application"""
        st.sidebar.markdown("### Menu")
        page = st.sidebar.selectbox(
            label="Selectionner une page", options=self.pages,
            format_func=lambda x: x["title"]
            )
        st.sidebar.markdown("___")
        page["name_page"]


def results_page(file_model= 'baselineModel'):
    """ The main page of the application """
    st.set_page_config(layout="wide")
    st.markdown("# Projet 2")
    st.markdown("### Proposition d'une application")
    st.markdown("**Créée par Charles, Amir et Moindzé**")

    st.markdown("---")

    st.markdown("### Résultats")

    with st.form(key="my_form"):
        
        sample = "trop de services qui se superposent à différents échelons départementales, régionales, nationales. Aucune économie possible ne peut se réaliser. Son utilité est nulle ou presque. Par exemple, l'intercommunalité serait à revoir, la commune perd son âme au détriment de tous ces nouveaux services et nos impôts augmentent pour financer cette structure"
        message_text = st.text_area(
            "Entrer la reponse à évaluer",
            sample,
            height=200,
            help=" Une reponse issue du dataset du grand débat "
        )

        submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            st.success("✅ Done!")


    if message_text != '':

        model = DissidentModelWrapper.load(file_model)

        result = model.sentensize_and_predict(message_text)

        df = pd.DataFrame(columns=["phrases", "predictions"])
    
        for j in range(len(result)):
            label = result[j][1]
            text = result[j][0]
            new_row = {"phrases" : text, "predictions" : label}
            df = df.append(new_row, ignore_index=True)
        
        st.markdown("##### Prédictions")
        st.table(df)

if __name__ == '__main__':
    app = Application()
    page = results_page(*sys.argv[1:])
    app.add_page("Resultats", page)
    app.run()

