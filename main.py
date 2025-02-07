import os
import streamlit as st
import concurrent.futures
import time
import datetime
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Configure Streamlit page
st.set_page_config(page_title="Chatbot", page_icon="🔍")

# Initialize session state for chat and memory
if "chat_open" not in st.session_state:
    st.session_state["chat_open"] = False
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(input_key="question", memory_key="history")

# Toggle chat window
if st.button("🤖 Chat with us"):
    st.session_state["chat_open"] = not st.session_state["chat_open"]

# Show chat UI if opened
if st.session_state["chat_open"]:
    st.title("🤖 QueryServe 🤖")
    st.subheader("How can I assist you today?")
    st.write("I can answer questions based on URLs or general knowledge.")

    # Input fields
    api_key = st.text_input("Enter your Google API Key", type="password")
    query = st.text_input("Ask a question")

    # URLs for content retrieval
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

    # Process query on button click
    if st.button("Get Answer"):
        if not api_key.strip():
            st.error("Please enter a valid Google API Key.")
        elif not query.strip():
            st.error("Please enter a query.")
        else:
            try:
                os.environ["GOOGLE_API_KEY"] = api_key.strip()
                st.write("🔄 Loading content from URLs...")

                # Load content from URLs in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    docs = list(executor.map(load_content, urls))

                # Flatten and split content
                docs_list = [item for sublist in docs for item in sublist]
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
                doc_splits = text_splitter.split_documents(docs_list)

                # FAISS index handling
                st.write("🔄 Updating FAISS index if needed...")
                temp_faiss_index = None
                last_updated_time = time.time()

                def update_faiss_index(docs_splits, embeddings_model):
                    global temp_faiss_index, last_updated_time
                    if temp_faiss_index is not None and (time.time() - last_updated_time) < 3600:
                        st.write("✅ FAISS index is up to date.")
                        return temp_faiss_index

                    st.write("🔄 Updating FAISS index...")
                    vectorstore = FAISS.from_documents(docs_splits, embeddings_model)
                    temp_faiss_index = vectorstore
                    last_updated_time = time.time()
                    return vectorstore

                embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vectorstore = update_faiss_index(doc_splits, embeddings_model)

                # Convert FAISS index to retriever
                retriever = vectorstore.as_retriever()

                # Retrieve relevant documents
                retrieved_docs = retriever.invoke(query)
                if retrieved_docs and any(doc.page_content.strip() for doc in retrieved_docs):
                    context = "\n".join([doc.page_content for doc in retrieved_docs])
                    response_source = "🔍 Based on retrieved documents."
                else:
                    context = "No relevant documents found."
                    response_source = "🧠 Answering based on general knowledge."

                st.write(response_source)

                # Initialize LLM with memory
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    system_message="Always begin your response with an appropriate greeting based on the time of day.",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                )

                # Define prompt with memory support
                after_rag_prompt = ChatPromptTemplate.from_template(
                    """Answer the question based only on the following context: {context}
                    First, greet the user appropriately based on the current time of day.
                    Previous conversation history:
                    {history}
                    Question: {question}
                    """
                )

                # Process query with memory
                after_rag_chain = (
                    {
                        "context": RunnablePassthrough().invoke(context),
                        "question": RunnablePassthrough().invoke(query),
                        "history": RunnablePassthrough().invoke(st.session_state["memory"].load_memory_variables({})),
                    }
                    | after_rag_prompt
                    | llm
                    | StrOutputParser()
                )

                # Get and display the answer
                answer = after_rag_chain.invoke(query)
                st.success(f"Answer: {answer}")

                # Store query and answer in memory
                st.session_state["memory"].save_context({"question": query}, {"answer": answer})

            except Exception as e:
                st.error(f"An error occurred: {e}")
