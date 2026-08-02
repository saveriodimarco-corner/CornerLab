import streamlit as st

from src.database.sqlite_utils import get_database_url, get_project_status, init_database
from src.utils.logging_config import configure_logging


configure_logging().info("Starting CornerLab Streamlit app")
init_database()

st.set_page_config(page_title="CornerLab PRO", page_icon="📊", layout="wide")

st.title("CornerLab PRO")
st.caption("Version 1.1")

st.subheader("Database status")
st.success(f"SQLite database ready at {get_database_url()}")

st.subheader("Project status")
st.info(get_project_status())
