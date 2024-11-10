# Chatbot for Querying Content from URLs Using Google Generative AI

This project is a Streamlit application that allows users to interact with a chatbot capable of answering questions based on content fetched from URLs. It utilizes Google's Generative AI model along with various tools from the LangChain ecosystem to load, embed, and query documents efficiently.

## Features
- **Time-based Greeting**: The chatbot greets the user based on the time of day (morning, afternoon, evening).
- **Content Loading from URLs**: Fetches content from predefined URLs using the `WebBaseLoader`.
- **Text Splitting**: Splits large documents into smaller chunks using `CharacterTextSplitter`.
- **FAISS Indexing**: Uses FAISS to index document chunks for efficient retrieval.
- **Generative AI for Answering**: Utilizes Google's Generative AI model to generate answers based on the content retrieved.
- **Parallel Processing**: Loads content from multiple URLs in parallel using `ThreadPoolExecutor` for improved performance.

## Installation

### Prerequisites

1. **Python 3.x** - Make sure Python 3 is installed on your system.
2. **Streamlit** - For building the web interface.
3. **LangChain** - For document processing and querying.
4. **Google API Key** - You need a Google API key for using the Generative AI model.

### Setup

Clone this repository or download the code to your local machine.

```bash
git clone <repository_url>
cd <repository_directory>
