# Chat PDF Application

Chat PDF Application is a Streamlit-based tool that allows users to interact with PDF documents through a chat interface. Users can upload multiple PDFs, extract and index the text, and then ask questions related to the content of these PDFs to receive detailed answers. The application also provides information about the source PDF(s) related to the answers.

## Features

- **Upload PDFs**: Allows users to upload multiple PDF files.
- **Text Extraction & Chunking**: Extracts text from PDFs, splits it into chunks, and indexes it using a vector database.
- **Question Answering**: Users can ask questions and get answers based on the content of the uploaded PDFs.
- **Source Identification**: Displays the source PDF(s) related to the answer and provides source information from metadata.

## Models and Tools Used

- **Google Generative AI**: Used for generating embeddings and answering questions. The specific model used for embeddings is `"models/embedding-001"` and the chat model is `"gemini-pro"`.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors, used to store and retrieve text chunks based on embeddings.
- **LangChain**: A framework used to manage language models and chain tasks together, including prompt generation and question answering.
- **Streamlit**: A framework for creating interactive web applications, used to build the user interface for uploading PDFs and interacting with the chat interface.
- **PyPDF2**: Used for extracting text from PDF files.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/bhargav0807/ChatWithPdfs.git
    cd ChatWithPdfs
    ```

2. **Create a Virtual Environment** (Optional)

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the Required Packages**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables**

    Create a `.env` file in the project root and add your Google API key:

    ```env
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. **Start the Streamlit Application**

    ```bash
    streamlit run app.py
    ```

    This will start the Streamlit server and open the application in your default web browser.

2. **Upload and Process PDFs**

    - Use the file uploader in the sidebar to upload your PDF files.
    - Click on "Submit & Process" to extract text, split it into chunks, and store it in a vector database.

3. **Ask Questions**

    - Enter your question in the text input box.
    - Press Enter to get an answer based on the content of the uploaded PDFs.
    - The application will display the source PDF(s) related to the answer and provide metadata from the vector store.

## Code Structure

- `app.py`: Main Streamlit application file.
- `requirements.txt`: Lists the Python packages required for the project.
- `.env`: Contains environment variables, such as API keys.

## Contact

For any questions or support, please contact [bhargavraj.ramagiri@gmail.com].
