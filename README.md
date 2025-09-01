# YouTube AI Video Assistant

An intelligent web application that uses AI to summarize YouTube videos and answer questions about their content. Simply provide a YouTube video URL, and the assistant will fetch the transcript, process it, and allow you to interact with the video's content through a user-friendly interface.

This project is built using Gradio for the UI, LangChain for orchestrating AI workflows, and IBM WatsonX for powerful language models.

## Features

-   **Automatic Summarization:** Get a concise, single-paragraph summary of any YouTube video with an available English transcript.
-   **Interactive Q&A:** Ask specific questions about the video's content and receive detailed answers based on the transcript.
-   **Simple Web Interface:** Easy-to-use interface built with Gradio, requiring no complex setup.
-   **Modular Codebase:** The project is structured into logical modules for YouTube utilities, text processing, and AI model handling, making it easy to maintain and extend.

## How It Works

The application follows a simple yet powerful workflow:

1.  **URL Input**: The user provides a YouTube video URL.
2.  **Transcript Fetching**: The application extracts the video ID and uses the `youtube-transcript-api` to fetch the English transcript.
3.  **Text Processing**: The raw transcript is formatted into clean text. For the Q&A feature, this text is split into smaller, overlapping chunks.
4.  **Vector Embedding**: The text chunks are converted into numerical representations (embeddings) using an IBM WatsonX embedding model. These embeddings are stored in a FAISS vector store for efficient similarity searching.
5.  **AI Interaction (LangChain & WatsonX)**:
    -   For **summarization**, the full transcript is sent to an IBM WatsonX LLM via a LangChain prompt to generate a summary.
    -   For **Q&A**, the user's question is used to search the vector store for the most relevant text chunks. This context, along with the question, is then sent to the LLM to generate an accurate answer.
6.  **Display**: The results are displayed back to the user in the Gradio web interface.

## Project Structure

The project is organized into a clean, modular structure for better readability and maintenance.