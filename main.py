import os  # Handle environment variables
import streamlit as st  # Create the web interface
from langchain_community.document_loaders import WebBaseLoader  # Load content from URLs
from langchain.vectorstores import FAISS  # FAISS for document retrieval
from langchain_huggingface import HuggingFaceEmbeddings  # Use HuggingFace embeddings
from langchain_core.runnables import RunnablePassthrough  # Pass data through the chain
from langchain_core.output_parsers import StrOutputParser  # Parse the output of the LLM
from langchain_core.prompts import ChatPromptTemplate  # Define prompt structure
from langchain.text_splitter import CharacterTextSplitter  # Split documents into chunks
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Generative AI
import concurrent.futures  # Parallel execution
import time  # Handle time-related functions

# Configure Streamlit page
st.set_page_config(page_title="Chatbot", page_icon="🤖")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store previous user queries and responses
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False  # Toggle chat UI

# Button to open/close the chat
if st.button("🤖 Chat with us"):
    st.session_state.chat_open = not st.session_state.chat_open

# Show chat UI if open
if st.session_state.chat_open:
    st.title("🤖 QueryServe 🤖")
    st.subheader("How can I assist you today?")
    st.write("I can answer questions based on content from URLs. Just provide your API key and query.")

    # Input for API key and user query
    api_key = st.text_input("Enter your Google API Key", type="password")
    query = st.text_input("Ask a question")

    # URLs to load content from
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

    # Run the chatbot when the user clicks "Get Answer"
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

                # Flatten document list
                docs_list = [item for sublist in docs for item in sublist]

                # Split documents into smaller chunks
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
                doc_splits = text_splitter.split_documents(docs_list)

                # FAISS index for document retrieval
                st.write("🔄 Updating FAISS index if needed...")
                temp_faiss_index = None
                last_updated_time = time.time()

                # Function to update FAISS index
                def update_faiss_index(docs_splits, embeddings_model):
                    global temp_faiss_index, last_updated_time
                    if temp_faiss_index is not None:
                        current_time = time.time()
                        if current_time - last_updated_time < 3600:
                            st.write("✅ FAISS index is up to date, skipping update.")
                            return temp_faiss_index
                        else:
                            st.write("🔄 1 hour has passed, updating FAISS index.")

                    vectorstore = FAISS.from_documents(docs_splits, embeddings_model)
                    temp_faiss_index = vectorstore
                    last_updated_time = time.time()
                    return vectorstore

                embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
                vectorstore = update_faiss_index(doc_splits, embeddings_model)

                # Convert FAISS index to retriever
                retriever = vectorstore.as_retriever()

                # Retrieve previous chat history
                chat_history_text = "\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state.chat_history])

                # Define prompt template (including chat history)
                after_rag_template = """Answer the question based only on the following context: {context} 
                    First, greet the user appropriately based on the current time of day.

                    Here is the previous conversation history to maintain context:
                    {chat_history}

                    Question: {question} 
                """
                
                after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

                # Create chain for processing user input
                after_rag_chain = (
                {"context": retriever, "question": RunnablePassthrough(), "chat_history": chat_history_text}
                | after_rag_prompt
                | ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    system_message="Always begin with an appropriate greeting based on the time of day.",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                )
                | StrOutputParser()
            )


                # Invoke the chain with user query
                answer = after_rag_chain.invoke({"context": retriever, "question": query, "chat_history": chat_history_text})

                # Store chat history
                st.session_state.chat_history.append((query, answer))

                # Display answer
                st.success(f"Answer: {answer}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
