import streamlit as st
import requests

st.set_page_config("Chat with Documents")
st.title("💬 Chat with Your PDFs")

# API_URL = "http://localhost:8000"
# API_URL = "http://localhost:8000"
API_URL = "http://backend:8000"



with st.sidebar:
    st.header("Upload PDFs")
    pdf_docs = st.file_uploader("Select up to 20 PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Uploading..."):
            files = [("files", (f.name, f.read(), f.type)) for f in pdf_docs]
            res = requests.post(f"{API_URL}/upload/", files=files)
            st.success("Uploaded and processed.")

st.header("Ask a question")
question = st.text_input("Your question")
if st.button("Get Answer"):
    if question:
        res = requests.get(f"{API_URL}/query/", params={"question": question})
        st.subheader("Answer:")
        st.write(res.json()["answer"])

# if st.button("View Metadata"):
#     res = requests.get(f"{API_URL}/metadata/")
#     st.json(res.json())
