import os  # Import os module to handle environment variables
import streamlit as st  # Import Streamlit for creating the web interface
from langchain_community.document_loaders import WebBaseLoader  # Import WebBaseLoader to load content from URLs
from langchain.vectorstores import FAISS  # Import FAISS to manage the vector store for document retrieval
from langchain_huggingface import HuggingFaceEmbeddings  # Import HuggingFaceEmbeddings for embedding documents
from langchain_core.runnables import RunnablePassthrough  # Import RunnablePassthrough for passing data through the chain
from langchain_core.output_parsers import StrOutputParser  # Import StrOutputParser to parse the output of the LLM
from langchain_core.prompts import ChatPromptTemplate  # Import ChatPromptTemplate to define the prompt structure
from langchain.text_splitter import CharacterTextSplitter  # Import CharacterTextSplitter to split documents into chunks
from langchain_google_genai import ChatGoogleGenerativeAI  # Import ChatGoogleGenerativeAI for Google's GenAI model
import concurrent.futures  # Import concurrent.futures for parallel execution
import time  # Import time for time-related functions
import datetime  # Import datetime to work with current date and time

# Function to get a greeting based on the time of day
def get_greeting():
    current_hour = datetime.datetime.now().hour  # Get the current hour of the day
    if current_hour < 12:  # Check if it's before noon
        return "Good Morning!"  # Return "Good Morning!" if before noon
    elif 12 <= current_hour < 18:  # Check if it's between noon and 6 PM
        return "Good Afternoon!"  # Return "Good Afternoon!" if in the afternoon
    else:  # Otherwise, it must be after 6 PM
        return "Good Evening!"  # Return "Good Evening!" if after 6 PM

# Configure the Streamlit page with a title and an icon
st.set_page_config(page_title="Chatbot", page_icon="🔍")

# Initialize session state for chat UI toggle
if 'chat_open' not in st.session_state:
    st.session_state['chat_open'] = False  # Initialize the chat UI as closed

# Bot icon button (emoji-based) to open/close the chat
if st.button("🤖 Chat with us"):  # Create a button to toggle chat UI
    st.session_state['chat_open'] = not st.session_state['chat_open']  # Toggle chat UI on button click

# Show the full chat UI only if the chat is open
if st.session_state['chat_open']:  # Check if the chat is open
    st.title("🤖 QueryServe 🤖")  # Display the chatbot title
    st.subheader(get_greeting() + " How can I assist you today?")  # Display greeting based on time of day
    st.write("I can help you answer questions based on content from URLs. Just provide me with the API key and your query.")  # Instruction for the user

    # Input fields for API key and user query
    api_key = st.text_input("Enter your Google API Key", type="password")  # Prompt the user for the Google API Key
    query = st.text_input("Ask a question")  # Prompt the user for a query

    # Define URLs to fetch content from
    urls = [
        "https://wordpress.com/",
        "https://wordpress.org/download/",
        "https://wordpress.org/",
        "https://en.wikipedia.org/wiki/WordPress",
        "https://apps.apple.com/us/app/wordpress-website-builder/id335703880",
        "https://play.google.com/store/apps/details?id=org.wordpress.android&hl=en_IN&pli=1",
    ]  # List of URLs from which the chatbot will fetch content

    # Function to load content from a single URL
    def load_content(url):
        return WebBaseLoader(url).load()  # Use WebBaseLoader to load content from the given URL

    # Run the chatbot when the user clicks "Get Answer"
    if st.button("Get Answer"):  # Check if the "Get Answer" button is clicked
        if not api_key.strip():  # If API key is not entered
            st.error("Please enter a valid Google API Key.")  # Show error message for missing API key
        elif not query.strip():  # If the query is not entered
            st.error("Please enter a query.")  # Show error message for missing query
        else:
            try:
                # Set the API key
                os.environ["GOOGLE_API_KEY"] = api_key.strip()  # Set the entered API key in the environment variables
                st.write("🔄 Loading content from URLs...")  # Notify the user that content is being loaded

                # Load content from URLs in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:  # Use ThreadPoolExecutor to load content in parallel
                    docs = list(executor.map(load_content, urls))  # Load content from all URLs in parallel

                # Flatten list of documents
                docs_list = [item for sublist in docs for item in sublist]  # Flatten the list of documents into a single list

                # Split the content into chunks
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)  # Initialize a text splitter
                doc_splits = text_splitter.split_documents(docs_list)  # Split the documents into chunks

                # Temporary FAISS index handling
                st.write("🔄 Updating FAISS index if needed...")  # Notify the user that the FAISS index is being updated
                temp_faiss_index = None  # Initialize a temporary FAISS index
                last_updated_time = time.time()  # Store the last update time

                # Function to update the FAISS index if needed
                def update_faiss_index(docs_splits, embeddings_model):
                    global temp_faiss_index, last_updated_time  # Use global variables for FAISS index and last update time
                    if temp_faiss_index is not None:  # Check if there is an existing FAISS index
                        current_time = time.time()  # Get the current time
                        if current_time - last_updated_time < 3600:  # Check if less than 1 hour has passed
                            st.write("✅ FAISS index is up to date, skipping update.")  # If up-to-date, skip update
                            return temp_faiss_index  # Return the current FAISS index
                        else:
                            st.write("🔄 1 hour has passed, updating FAISS index.")  # If more than 1 hour has passed, update the index
                    
                    # If the index needs to be updated, create a new FAISS index
                    vectorstore = FAISS.from_documents(docs_splits, embeddings_model)  # Create a new FAISS index with the document splits
                    temp_faiss_index = vectorstore  # Update the temporary FAISS index
                    last_updated_time = time.time()  # Update the last update time
                    return vectorstore  # Return the updated FAISS index

                embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')  # Load the embedding model
                vectorstore = update_faiss_index(doc_splits, embeddings_model)  # Update or fetch the FAISS index

                # Convert to retriever for querying
                retriever = vectorstore.as_retriever()  # Convert the FAISS index to a retriever for querying

                # Setup for Chatbot
                st.write("🔄 Generating answer...")  # Notify the user that the answer is being generated
                llm = ChatGoogleGenerativeAI(  # Initialize the Google Generative AI model
                    model="gemini-1.5-flash",  # Specify the model to use
                    temperature=0,  # Set the temperature for the response (lower is more deterministic)
                    max_tokens=None,  # No limit on the number of tokens in the response
                    timeout=None,  # No timeout for the request
                    max_retries=2,  # Retry up to 2 times in case of failure
                )

                # Define the prompt template and create a LangChain chain
                after_rag_template = """Answer the question based only on the following context: {context} Question: {question} """  # Define the prompt template
                after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)  # Create the prompt from the template
                after_rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | after_rag_prompt | llm | StrOutputParser())  # Define the chain to process the query

                # Run the chain and display the result
                answer = after_rag_chain.invoke(query)  # Run the chain with the user query
                st.success(f"Answer: {answer}")  # Display the answer to the user

            except Exception as e:  # Handle any exceptions
                st.error(f"An error occurred: {e}")  # Show an error message if something goes wrong
