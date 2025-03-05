<h1 align="center">📡 Chat With Multiple PDFs 📚</h1>

---

<h2 align="left">➡️ Description</h2>

This project provides an intuitive interface to interact with multiple PDF documents. By leveraging advanced language models and vector databases, users can ask questions and receive contextually relevant answers based on the content of the uploaded PDFs.

<h2 align="left">🚀 Features</h2>

- **PDF Content Retrieval**: Upload and process multiple PDF documents to extract their textual content.
- **AI-Powered Chat**: Utilize advanced AI models to generate responses based on the content of the PDFs.
- **User-Friendly Interface**: Interact with the system through an intuitive web interface.

<h2 align="left">⌛ How It Works</h2>

1. **Upload PDFs**: Users can upload multiple PDF files through the web interface.
2. **Content Extraction**: The application reads and extracts text from the uploaded PDFs.
3. **Text Chunking**: Extracted text is divided into manageable chunks to improve search accuracy.
4. **Vector Storage**: Text chunks are embedded using AI models and stored in a vector database for efficient retrieval.
5. **Conversational Interaction**: Users can ask questions, and the system retrieves relevant information from the PDFs to generate accurate responses.

<h2 align="left">🛠️ Setup Instructions</h2>

1. Clone the repository:
   ```sh
   git clone https://github.com/MrGenius18/Chat_With_Multiple_pdfs.git
   cd Chat_With_Multiple_pdfs
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   
4. Set Up API Keys:
   Create a .env file in the root directory and add your API keys:
   ```sh
   GOOGLE_API_KEY=your_api_key
   ```

<h2 align="left">♻️ Usage</h2>

1. Run the application:
   ```sh
   streamlit run app.py
   ```
   Access the application via the URL provided by Streamlit, typically http://localhost:8501
2. Upload PDF files through the interface.
3. Ask questions and get responses based on PDF content.

<h2 align="left">✨ Demo</h2>

![Chat wwith Multi PDFs UI Demo Screenshot](https://github.com/MrGenius18/Chat_With_Multiple_pdfs/blob/800821a6f7d91a88cfdc88e8536ebe2e50758b9b/Demo.png)

<h2 align="left">⭐ Acknowledgments</h2>

This project utilizes the following technologies:

- **PdfPlumber** for PDF text extraction
- **Streamlit**: An open-source app framework for Machine Learning and Data Science teams.
- **LangChain**: A framework for developing applications powered by language models.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.

<h2 align="left">🤝 Contributing</h2>

Contributions are welcome! Feel free to fork this repository, create a branch, and submit a pull request.

<h2 align="left">🗳️ License</h2>

This project is licensed under the MIT License.

---
