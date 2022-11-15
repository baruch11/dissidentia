
try:
    import streamlit as st
except ModuleNotFoundError:
    raise (
        """
        Please install streamlit
        """
    )

from home_page import home_page
from results_page import results_page


class Application:
    
    def __init__(self) -> None:
        self.pages = []
    
    def add_page(self, title, name_page):

        self.pages.append({"title": title, "name_page" : name_page})

    def run(self):
        st.set_page_config(layout="wide")
        st.sidebar.markdown("### Menu")
        page = st.sidebar.selectbox(
            label="Selectionner une page", options=self.pages,
            format_func=lambda x: x["title"]
            )
        st.sidebar.markdown("___")
        page["name_page"]()
    
if __name__ == '__main__':
    app = Application()
    app.add_page("Home Page", home_page)
    app.add_page("Resultats", results_page)
    app.run()