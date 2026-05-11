import streamlit as st
import chromadb
from pathlib import Path
import sys

# Ensure the core module is accessible
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Anchor the database to the data folder
DATA_DIR = BASE_DIR / "data"
CHROMA_PATH = str(DATA_DIR / "chroma_db")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="employee_voices_v2",
    metadata={"hnsw:space": "cosine"}
)

@st.dialog("Confirm Deletion")
def confirm_delete_dialog(emp_id, emp_name):
    st.warning(f"Are you sure you want to permanently delete the biometric profile for **{emp_name}**?")
    if st.button("Yes, Delete Profile", type="primary", use_container_width=True):
        collection.delete(ids=[emp_id])
        st.success(f"Deleted {emp_name} from database.")
        st.rerun()

st.set_page_config(page_title="Database Manager", page_icon="⚙️", layout="wide")

st.title("⚙️ Database Administration Dashboard")
st.markdown("Manage enrolled employee biometric profiles and system database entries.")

# Fetch all entries
data = collection.get(include=['metadatas'])
ids = data['ids']
metadatas = data['metadatas']

total_enrolled = len(ids)
st.metric("Total Enrolled Employees in Database", total_enrolled)

st.divider()

if total_enrolled == 0:
    st.info("No employees enrolled in the database yet.")
else:
    # Header Row
    cols = st.columns([0.5, 2, 2, 2])
    cols[0].markdown("**No.**")
    cols[1].markdown("**Employee Name**")
    cols[2].markdown("**Job Role**")
    cols[3].markdown("**Actions**")
    
    st.divider()

    # Data Rows
    for i, (emp_id, meta) in enumerate(zip(ids, metadatas)):
        name = meta.get("name", "Unknown")
        role = meta.get("role", "Unknown")
        
        row_cols = st.columns([0.5, 2, 2, 2])
        row_cols[0].write(f"{i+1}")
        row_cols[1].write(name)
        row_cols[2].write(role)
        
        # Actions
        action_cols = row_cols[3].columns(2)
        
        # Update Button
        if action_cols[0].button("Update", key=f"upd_{emp_id}", use_container_width=True):
            st.session_state['edit_name'] = name
            st.session_state['edit_role'] = role
            st.session_state['edit_original_id'] = emp_id
            st.switch_page("pages/1_Voice_Enrollment.py")
            
        # Delete Button
        if action_cols[1].button("Delete", key=f"del_{emp_id}", type="secondary", use_container_width=True):
            confirm_delete_dialog(emp_id, name)

st.sidebar.markdown("---")
st.sidebar.info("This dashboard provides administrative access to the ChromaDB vector store.")
