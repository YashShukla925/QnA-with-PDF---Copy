import streamlit as st
import requests

st.set_page_config("Chat with Documents")
st.title("💬 Multilingual PDF QA")

API_URL = "http://localhost:8000"

# Sidebar for uploading PDFs
with st.sidebar:
    st.header("Upload PDFs")
    pdf_docs = st.file_uploader("Select up to 20 PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Uploading..."):
                files = [("files", (f.name, f.read(), f.type)) for f in pdf_docs]
                res = requests.post(f"{API_URL}/upload/", files=files)
                st.success("Uploaded and processed.")

# Ask question
st.header("Ask a question")
question = st.text_input("Your question in English")

if st.button("Get Answer"):
    if question:
        res = requests.get(f"{API_URL}/query/", params={"question": question})
        data = res.json()
        
        st.subheader("Answer:")
        st.write(data["answer"])

        st.subheader("Sources:")
# Show only the first 2 sources
        for idx, src in enumerate(data["sources"][:2]):
            file_name, page = src["file"], src["page"]
            
            # Display the source text
            st.markdown(f"📄 **{file_name} - Page {page}**")
            
            # Clickable link to open PDF at that page
            # pdf_url = f"{API_URL}/static/{file_name}#page={page}"
            # st.markdown(f"[📖 Go to Page {page}]({pdf_url})", unsafe_allow_html=True)


