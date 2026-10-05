import streamlit as st
import sys
sys.path.append("src")
from qa_pipeline import ask

st.title("Document Q&A System")
st.write("Ask questions grounded in the indexed documents.")

sample_questions = [
    "What is Azure used for?",
    "What is the FlutterFlirt ERP scenario about?",
]

col1, col2 = st.columns(2)
for i, sq in enumerate(sample_questions):
    if (col1 if i % 2 == 0 else col2).button(sq):
        st.session_state.query = sq

query = st.text_input("Your question:", value=st.session_state.get("query", ""))

if st.button("Ask") and query:
    with st.spinner("Retrieving and generating answer..."):
        answer, sources = ask(query)
    st.markdown(f"### Answer\n{answer}")
    with st.expander("Sources used"):
        for i, s in enumerate(sources):
            st.markdown(f"**[{i+1}]** {s}")