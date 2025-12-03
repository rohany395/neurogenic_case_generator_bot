# Fix for SQLite version issue with ChromaDB - must be before any chromadb import
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # pysqlite3 not available, use system sqlite3

import streamlit as st


pg=st.navigation([st.Page("./src/main.py",title="Neurogenic Case Generator")])
st.set_page_config(page_title="Case generator bot",)
pg.run()