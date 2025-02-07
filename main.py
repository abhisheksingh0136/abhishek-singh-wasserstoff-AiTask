import os  # Handle environment variables
import streamlit as st  # Streamlit for UI
from langchain_community.document_loaders import WebBaseLoader  # Load content from URLs
from langchain.vectorstores import FAISS  # FAISS for document retrieval
from langchain_huggingface import HuggingFaceEmbeddings  # Embedding model
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's GenAI model
from langchain.memory import ConversationBufferMemory  # Memory for chat history
from langchain.chains import ConversationalRetrievalChain  # Conversational RAG chain
import concurrent.futures  # Parallel execution
import time  # For timestamps

# Configure Streamlit page
st.set_page_config(page_title="Chatbot", page_icon="🤖")

# Initialize session state for chat UI and memory
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store chat history

# Bot toggle button
if st.button("🤖 Chat with us"):
    st.session_state.chat_open = not st.session_state.chat_open

# Show chatbot UI only when toggled open
if st.session_state.chat_open:
    st.title("🤖 QueryServe 🤖")
    st.subheader("How can I assist you today?")
    st.write("I can answer questions based on web content. Provide your API key and query.")

    # Input fields
    api_key = st.text_input("Enter your Google API Key", type="password")
    query = st.text_input("Ask a question")

    # URLs to fetch content from
    urls = [
        "https://wordpress.com/",
        "https://wordpress.org/download/",
        "https://wordpress.org/",
        "https://en.wikipedia.org/wiki/WordPress",
        "https://apps.apple.com/us/app/wordpress-website-builder/id335703880",
        "https://play.google.com/store/apps/details?id=org.wordpress.android&hl=en_IN&pli=1",
    ]

    # Function to load content from a URL
    def load_content(url):
        return WebBaseLoader(url).load()

    # Execute chatbot logic when user clicks "Get Answer"
    if st.button("Get Answer"):
        if not api_key.strip():
            st.error("Please enter a valid Google API Key.")
        elif not query.strip():
            st.error("Please enter a query.")
        else:
            try:
                # Set API key
                os.environ["GOOGLE_API_KEY"] = api_key.strip()
                st.write("🔄 Loading content from URLs...")

                # Load content from URLs in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    docs = list(executor.map(load_content, urls))

                # Flatten list of documents
                docs_list = [item for sublist in docs for item in sublist]

                # Split content into chunks
                from langchain.text_splitter import CharacterTextSplitter
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
                doc_splits = text_splitter.split_documents(docs_list)

                # Update FAISS index
                st.write("🔄 Updating FAISS index if needed...")
                embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
                vectorstore = FAISS.from_documents(doc_splits, embeddings_model)

                # Setup memory for chat history
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

                # Create a conversational retrieval chain
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        system_message="Always begin your response with an appropriate greeting based on the time of day. If the user explicitly asks for a greeting, respond with an appropriate wish.",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    ),
                    retriever=vectorstore.as_retriever(),
                    memory=memory,
                    return_source_documents=True
                )

                # Run the query through the chatbot with chat history
                response = qa_chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
                answer = response["answer"]

                # Store conversation in session state
                st.session_state.chat_history.append((query, answer))

                # Display answer
                st.success(f"**Answer:** {answer}")

                # Show chat history
                if st.session_state.chat_history:
                    st.subheader("Chat History")
                    for idx, (q, a) in enumerate(st.session_state.chat_history):
                        st.write(f"**Q{idx+1}:** {q}")
                        st.write(f"**A{idx+1}:** {a}")
                        st.write("---")

            except Exception as e:
                st.error(f"An error occurred: {e}")
