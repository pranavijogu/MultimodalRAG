# frontend.py

import streamlit as st
import requests
import os

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000"  # URL of your FastAPI backend

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal RAG Chatbot", layout="wide")

st.title("🧠 Multimodal RAG Chatbot")
st.markdown("""
Welcome! This chatbot can answer questions about uploaded PDFs (containing text and images) or audio files.
1.  **Upload a file** using the sidebar.
2.  **Wait for processing** to complete.
3.  **Ask a question** in the chat box below.
""")

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a PDF or Audio file",
        type=['pdf', 'mp3', 'wav', 'm4a']
    )

    if uploaded_file is not None:
        # Check if this is a different file than what was previously processed
        current_file_info = (uploaded_file.name, uploaded_file.size)
        previous_file_info = st.session_state.get("processed_file_info")
        
        if current_file_info != previous_file_info:
            st.session_state.processed_file_info = current_file_info
            file_name = uploaded_file.name
            file_type = uploaded_file.type
            
            # Determine the correct endpoint based on file type
            if "pdf" in file_type:
                endpoint = "/upload_pdf/"
            else:
                endpoint = "/upload_audio/"

            with st.spinner(f"Processing {file_name}..."):
                try:
                    # Prepare the file for the POST request
                    files = {'file': (file_name, uploaded_file.getvalue(), file_type)}
                    
                    # Send the file to the backend
                    response = requests.post(f"{BACKEND_URL}{endpoint}", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"✅ Successfully processed and indexed '{file_name}'")
                        # Force refresh of document list
                        if 'document_list' in st.session_state:
                            del st.session_state['document_list']
                    else:
                        st.error(f"❌ Error processing file: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: Could not connect to the backend at {BACKEND_URL}. Is it running?")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # --- Show Stored Documents Section ---
    st.header("📚 Stored Documents")
    
    # Add refresh button
    if st.button("🔄 Refresh Document List"):
        if 'document_list' in st.session_state:
            del st.session_state['document_list']
    
    # Get and cache document list
    if 'document_list' not in st.session_state:
        try:
            response = requests.get(f"{BACKEND_URL}/list_documents/")
            if response.status_code == 200:
                doc_data = response.json()
                st.session_state['document_list'] = doc_data
            else:
                st.session_state['document_list'] = {"documents": [], "total_chunks": 0}
        except:
            st.session_state['document_list'] = {"documents": [], "total_chunks": 0}
    
    doc_data = st.session_state['document_list']
    documents = doc_data.get('documents', [])
    total_chunks = doc_data.get('total_chunks', 0)
    
    if documents:
        st.success(f"📄 **{len(documents)} documents** stored ({total_chunks} chunks)")
        
        # Show document list
        for i, doc in enumerate(documents, 1):
            # Create a container for each document
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Show document name with icon
                    if doc.endswith('.pdf'):
                        icon = "📄"
                    elif doc.endswith(('.mp3', '.wav', '.m4a')):
                        icon = "🎵"
                    else:
                        icon = "📁"
                    st.write(f"{icon} **{doc}**")
                
                with col2:
                    # Delete button for each document
                    if st.button("🗑️", key=f"delete_{i}", help=f"Delete {doc}"):
                        try:
                            delete_response = requests.delete(f"{BACKEND_URL}/delete_document/{doc}")
                            if delete_response.status_code == 200:
                                st.success(f"✅ Deleted '{doc}'")
                                # Refresh document list
                                del st.session_state['document_list']
                                st.rerun()
                            else:
                                st.error(f"❌ Error deleting '{doc}'")
                        except Exception as e:
                            st.error(f"Error: {e}")
    else:
        st.info("📭 No documents uploaded yet")
    
    # --- Clear All Documents Button ---
    if documents:
        st.markdown("---")
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                try:
                    response = requests.delete(f"{BACKEND_URL}/clear_all/")
                    if response.status_code == 200:
                        st.success("✅ All documents cleared!")
                        del st.session_state['document_list']
                        st.session_state['confirm_clear'] = False
                        st.rerun()
                    else:
                        st.error("❌ Error clearing documents")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.session_state['confirm_clear'] = True
                st.warning("⚠️ Click again to confirm deletion of ALL documents")

# --- Main Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your documents?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Send the query to the backend
                response = requests.post(f"{BACKEND_URL}/query/", data={"user_question": prompt})
                
                if response.status_code == 200:
                    full_response = response.json().get("answer", "Sorry, I couldn't find an answer.")
                else:
                    full_response = f"Error from backend: {response.text}"
            
            except requests.exceptions.RequestException as e:
                full_response = f"Connection error: Could not connect to the backend at {BACKEND_URL}."
            except Exception as e:
                full_response = f"An unexpected error occurred: {e}"

        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
