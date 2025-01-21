
---

# Fusion RAG App

## Overview

Fusion RAG App is a Streamlit-based application that integrates Retrieval-Augmented Generation (RAG) with Reciprocal Rank Fusion (RRF) and LangChain. It allows users to query indexed data and receive responses based on the context provided, with the ability to evaluate the relevance of answers using Athina AI.

## Project Structure

```plaintext
fusion_rag_app/
├── app.py                 # Main Streamlit application logic
├── utils.py               # Helper functions (e.g., RRF, query generation)
├── data/                  # Data storage
│   └── context.csv        # Context data for queries
└── prompts/               # Prompt storage
    └── rag-fusion-query-generation.json  # LangSmith prompt
```

### Files Overview

- **app.py**: Main app file with the Streamlit logic, including data indexing, API keys management, and user query handling.
- **utils.py**: Contains helper functions for Reciprocal Rank Fusion (RRF) and query generation using LangSmith.
- **context.csv**: A CSV file that contains the context data used for querying.
- **rag-fusion-query-generation.json**: LangSmith query generation prompt configuration.

## Requirements

Before running the app, ensure that you have the following Python packages installed:

```bash
pip install streamlit langchain langchain-openai qdrant-client pandas datasets athina-ai langsmith
```

## Setup

1. **Clone the Repository**:
   If you haven't already, clone this repository to your local machine:

   ```bash
   git clone <repository_url>
   cd fusion_rag_app
   ```

2. **API Keys**:
   You need the following API keys:
   - OpenAI API Key
   - Athina API Key
   - Qdrant API Key

   You can provide these keys via the Streamlit secrets management (`st.secrets`) or input them directly into the app.

   Example secrets configuration:

   ```plaintext
   secrets/
   └── secrets.toml
   ```

   ```toml
   [general]
   OPENAI_API_KEY = "your_openai_api_key"
   ATHINA_API_KEY = "your_athina_api_key"
   QDRANT_API_KEY = "your_qdrant_api_key"
   ```

3. **Context Data**:
   Place your `context.csv` file in the `./data/` folder. This file should contain the data that will be used as the context for the queries.

4. **LangSmith Prompt**:
   Download the LangSmith prompt configuration (`rag-fusion-query-generation.json`) and place it in the `./prompts/` folder.

5. **Qdrant Setup**:
   Ensure that you have a Qdrant instance running and provide its URL in the app.

## Running the App

1. Navigate to the `fusion_rag_app` folder in your terminal.

2. Run the following command:

   ```bash
   streamlit run app.py
   ```

3. Open the Streamlit app in your browser by following the URL displayed in the terminal.

## Usage

Once the app is running:

1. **Enter a Query**: Type your query in the input box on the Streamlit interface.
2. **View Response**: The app will display a response generated based on the context.
3. **Context Display**: Expand the section to view the context used to generate the answer.
4. **Ground Truth Evaluation**: Optionally, you can input the ground truth for evaluating the relevance of the answer using Athina AI.

## Key Features

- **RAG Fusion**: Combines Retrieval-Augmented Generation and Reciprocal Rank Fusion to generate more accurate responses.
- **Streamlit UI**: Provides an easy-to-use interface for querying and evaluating responses.
- **API Key Management**: Streamlit's secrets management to securely handle API keys.
- **Answer Evaluation**: Uses Athina AI for relevance evaluation of responses.

## Troubleshooting

- **Missing API Keys**: If the API keys are not provided, the app will stop and display a warning.
- **Context Not Found**: If no relevant context is found for a query, the app will return a "don't know" response.

## Contributions

Feel free to contribute to this project by submitting issues or pull requests. If you have suggestions or improvements, they are always welcome!

---

Let me know if you need any further modifications or additional information in the README!
