import numpy as np
import streamlit as st
import pandas as pd
from nltk import tokenize


# To be replace by the model loading
class model:
    def sentencize_and_predict(self, text):
        sentences = tokenize.sent_tokenize(text)
        predictions = np.random.choice(['True','False'], size=len(sentences))
        return list(zip(sentences, predictions))


def results_page():
    st.markdown("# Projet 2")
    st.markdown("### Proposition d'une application")
    st.markdown("**Créée par Charles, Amir et Moindzé**")

    st.markdown("---")

    st.markdown("### Résultats")



    with st.form(key="my_form"):
        

        sample = "trop de services qui se superposent à différents échelons départementales, régionales, nationales. Aucune économie possible ne peut se réaliser. Par exemple, l'intercommunalité serait à revoir, la commune perd son âme au détriment de tous ces nouveaux services et nos impôts augmentent pour financer cette structure"
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

        result = model().sentencize_and_predict(message_text)

        df = pd.DataFrame(columns=["phrases", "predictions"])
    
        for j in range(len(result)):
            label = result[j][1]
            text = result[j][0]
            new_row = {"phrases" : text, "predictions" : label}
            df = df.append(new_row, ignore_index=True)
        
        st.markdown("#### Vérification des resultas")

        pred = st.checkbox('Prédictions')
        if pred:
            st.write(df)

                    