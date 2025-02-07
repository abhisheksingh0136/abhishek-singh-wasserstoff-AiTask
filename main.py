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
import concurrent.futures  
from langchain.memory import ConversationBufferMemory  # Import memory for chat history

# Set up Streamlit
st.set_page_config(page_title="Chatbot", page_icon="🔍")

# Initialize session state for chat memory
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize chat UI
if 'chat_open' not in st.session_state:
    st.session_state['chat_open'] = False

if st.button("🤖 Chat with us"):
    st.session_state['chat_open'] = not st.session_state['chat_open']

if st.session_state['chat_open']:
    st.title("🤖 QueryServe 🤖")  
    st.subheader("How can I assist you today?")
    st.write("I can help you answer questions based on content from URLs. Just provide me with the API key and your query.")

    # Input fields
    api_key = st.text_input("Enter your Google API Key", type="password")  
    query = st.text_input("Ask a question")  

    # Define URLs to fetch content from
    urls = [
        "https://wordpress.com/",
        "https://wordpress.org/download/",
        "https://wordpress.org/",
        "https://en.wikipedia.org/wiki/WordPress",
        "https://apps.apple.com/us/app/wordpress-website-builder/id335703880",
        "https://play.google.com/store/apps/details?id=org.wordpress.android&hl=en_IN&pli=1",
    ]

    # Load content from URLs
    def load_content(url):
        return WebBaseLoader(url).load()  

    if st.button("Get Answer"):  
        if not api_key.strip():
            st.error("Please enter a valid Google API Key.")  
        elif not query.strip():  
            st.error("Please enter a query.")  
        else:
            try:
                os.environ["GOOGLE_API_KEY"] = api_key.strip()  
                st.write("🔄 Loading content from URLs...")  

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    docs = list(executor.map(load_content, urls))  

                docs_list = [item for sublist in docs for item in sublist]  
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)  
                doc_splits = text_splitter.split_documents(docs_list)  

                # Update FAISS index
                embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')  
                vectorstore = FAISS.from_documents(doc_splits, embeddings_model)  
                retriever = vectorstore.as_retriever()  

                # Retrieve documents
                retrieved_docs = retriever.invoke(query)  
                if retrieved_docs:
                    context = "\n".join([doc.page_content for doc in retrieved_docs])  
                else:
                    context = ""  

                # Use memory buffer for conversation history
                chat_history = st.session_state['memory'].load_memory_variables({})["chat_history"]

                # Modified prompt with memory
                chat_prompt_template = """Use the following chat history and document context (if available) to answer the question.
                If no relevant context is found, use your own knowledge.
                Chat History:
                {chat_history}

                Document Context:
                {context}

                Question: {question}
                """
                chat_prompt = ChatPromptTemplate.from_template(chat_prompt_template)  

                # Format the prompt and pass it to the model
                formatted_prompt = chat_prompt.format(chat_history=chat_history, context=context, question=query)
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None, max_retries=2)

                # Process response
                answer = llm.invoke(formatted_prompt)  # Pass the formatted string directly to the model
                st.success(f"Answer: {answer}")  

                # Store interaction in memory
                st.session_state['memory'].save_context({"input": query}, {"output": answer})

                # Follow-up question generation
                follow_up_prompt_template = """Based on the chat history and answer, suggest 3-5 follow-up questions.
                Chat History:
                {chat_history}
                User Question: {question}
                AI Answer: {answer}
                """
                follow_up_prompt = ChatPromptTemplate.from_template(follow_up_prompt_template)  
                follow_up_chain = follow_up_prompt | llm | StrOutputParser()  
                follow_up_questions = follow_up_chain.invoke({"chat_history": chat_history, "question": query, "answer": answer})  

                if follow_up_questions:
                    st.subheader("💡 Follow-up Questions:")
                    for q in follow_up_questions.split("\n"):
                        if q.strip():
                            if st.button(q.strip()):  
                                query = q.strip()  
                                st.experimental_rerun()  

            except Exception as e:  
                st.error(f"An error occurred: {e}")
