import streamlit as st


pg=st.navigation([st.Page("./src/main.py",title="Neurogrnic Case Generator")])
st.set_page_config(page_title="Case generator bot",)
pg.run()