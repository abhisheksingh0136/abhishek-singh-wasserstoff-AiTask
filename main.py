import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory  # Importing the ConversationBufferMemory
import concurrent.futures
import time
import datetime

# Set up the Streamlit page
st.set_page_config(page_title="Chatbot", page_icon="🤖")

# Initialize session state
if 'chat_open' not in st.session_state:
    st.session_state['chat_open'] = False  # Whether chat is open

# Initialize session state for storing chat history in memory
if 'chat_memory' not in st.session_state:
    st.session_state['chat_memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Buffer memory

# Button to toggle chat visibility
if st.button("🤖 Chat with us"):
    st.session_state['chat_open'] = not st.session_state['chat_open']

# Display chat interface when the chat is open
if st.session_state['chat_open']:
    st.title("🤖 QueryServe 🤖")
    st.subheader("How can I assist you today?")
    st.write("I can help you answer questions based on content from URLs. Just provide me with the API key and your query.")
    
    # API Key and user query input fields
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

    # Run chatbot on "Get Answer" button click
    if st.button("Get Answer"):
        if not api_key.strip():
            st.error("Please enter a valid Google API Key.")
        elif not query.strip():
            st.error("Please enter a query.")
        else:
            try:
                # Set the API key
                os.environ["GOOGLE_API_KEY"] = api_key.strip()
                st.write("🔄 Loading content from URLs...")

                # Load content from URLs in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    docs = list(executor.map(load_content, urls))

                # Flatten the list of documents
                docs_list = [item for sublist in docs for item in sublist]

                # Split the content into chunks
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
                doc_splits = text_splitter.split_documents(docs_list)

                # FAISS index management
                embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
                vectorstore = FAISS.from_documents(doc_splits, embeddings_model)

                retriever = vectorstore.as_retriever()

                # Initialize the Google Generative AI model
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    system_message="Always begin your response with an appropriate greeting based on the time of day. If the user explicitly asks for a greeting, respond with an appropriate wish.",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                )

                # Create the prompt to include conversation history from the memory buffer
                chat_history = st.session_state['chat_memory'].buffer  # Retrieve chat history
                after_rag_template = f"""
                    Answer the question based only on the following context: {{context}} 
                    First, greet the user appropriately based on the current time of day.
                    Previous conversation history:
                    {chat_history}
                    Question: {{question}}
                """

                # Define the prompt and chain
                after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
                after_rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | after_rag_prompt | llm | StrOutputParser())

                # Run the chain and get the response
                answer = after_rag_chain.invoke(query)

                st.success(f"Answer: {answer}")

                # Store the conversation in memory buffer
                st.session_state['chat_memory'].add_user_message(query)  # Add user query to memory
                st.session_state['chat_memory'].add_ai_message(answer)  # Add bot response to memory

            except Exception as e:
                st.error(f"An error occurred: {e}")
