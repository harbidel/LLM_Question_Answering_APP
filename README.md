# LLM Question-Answering Application

![Header Image](https://github.com/harbidel/LLM_Question_Answering_APP/blob/main/img.jpg)

This Streamlit application utilizes OpenAI's GPT-3.5-turbo model for question-answering based on document embeddings. It allows users to upload documents in PDF, DOCX, or TXT format, chunks the document, and generates embeddings for each chunk. Users can then ask questions related to the document content, and the application provides answers using the GPT-3.5-turbo model.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/harbidel/LLM_Question_Answering_APP.git
    ```

2. Navigate to the project directory:

    ```bash
    cd chat_with_documents
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root with your OpenAI API key:

    ```
    OPENAI_API_KEY=your_api_key
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run chat_with_documents.py
    ```

2. Access the application in your browser.

3. Enter your OpenAI API key in the sidebar.

4. Upload a document (PDF, DOCX, or TXT) and configure chunk size and `k` parameter.

5. Click "Add Data" to process the document and create embeddings.

6. Ask questions in the text input, and the application will provide answers based on the document content.

## Demo 1

![Demo](https://github.com/harbidel/LLM_Question_Answering_APP/blob/main/Chat%20Doc1.gif)

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
