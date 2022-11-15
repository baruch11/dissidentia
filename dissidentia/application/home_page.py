import streamlit as st

def home_page():

    st.markdown("# Projet 2")
    st.markdown("### Proposition d'une application")
    st.markdown("**Créée par Charles, Amir et Moindzé**")

    st.markdown("---")

    st.markdown(
        """
        ### Description
        
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Annotation guidelines
        On cherche à détecter les critiques outrancières du pouvoir en place dans les réponses au grand débat. 
        On labellise chaque phrase indépendamment.
        On qualifie chaque phrase de critique outrancière sur 2 critères : 
        1) véhémence de la critique
        2) cible de la critique.
        
        ##### Véhémence
        Rentre dans ce critère tout ce qui est outrancier, haineux, ou calomnieux (c-a-d sans preuve évidente, juste sur la base de on-dit). Les phrases critique mais relativement neutre ne rentre pas dans ce cadre 
        
        **exemples :**
        "Tous pourris !" (outrance), "Les fonctionnaires ne recrutent que par copinage" (grosse généralité sans preuve)
        
        **contre-exemples :**
        "Il faut réduire le mille feuille administratif",
        "Il faut simplifier le système", 
        "A mon avis les fonctionnaires sont trop nombreux" (critique pas forcement justifiée mais pas outrancière)
        
        ##### cible
        Rentre dans ce critère tout ce qui est dirigé contre le pouvoir en place, c'est-à-dire l'Etat, les élus et toute l'administration (fonctionnaires, députés, ministres etc.)
        
        **exemples :** "Les fonctionnaires sont des feignasses", "Les hauts fonctionnaires sont corrompus"
        
        **contre-exemples :** "Les chômeurs sont des feignasses", "Il faut dégager les immigrés"
        """
    )

    st.markdown("---")