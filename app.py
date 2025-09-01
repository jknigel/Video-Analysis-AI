# app.py

import gradio as gr
from llm_handler import get_llm_model, get_embedding_model, create_vector_store, create_summarization_chain, create_qa_chain
from text_processor import chunk_text
from utils import get_video_id, get_transcript, format_transcript_as_text

# --- Initialize models once to improve performance ---
llm = get_llm_model()
embeddings = get_embedding_model()
summarization_chain = create_summarization_chain(llm)
qa_chain = create_qa_chain(llm)

# --- Global state to hold video data ---
# Using a simple dictionary for state management in this example
video_data_store = {
    "video_id": None,
    "transcript": "",
    "vector_store": None
}

def process_video_and_get_summary(video_url: str):
    """Processes a video URL to fetch transcript, create a vector store, and return a summary."""
    if not video_url:
        return "Please enter a YouTube URL.", "", ""

    video_id = get_video_id(video_url)
    if not video_id:
        return "Invalid YouTube URL provided.", "", ""

    # Avoid reprocessing the same video
    if video_data_store["video_id"] == video_id:
        summary = summarization_chain.run(transcript=video_data_store["transcript"])
        return f"Video '{video_id}' is already processed. Ready for Q&A.", summary, ""

    # Process new video
    raw_transcript = get_transcript(video_id)
    if not raw_transcript:
        return f"Could not retrieve a transcript for video '{video_id}'.", "", ""

    transcript_text = format_transcript_as_text(raw_transcript)
    chunks = chunk_text(transcript_text)
    vector_store = create_vector_store(chunks, embeddings)

    # Update global state
    video_data_store["video_id"] = video_id
    video_data_store["transcript"] = transcript_text
    video_data_store["vector_store"] = vector_store

    # Generate initial summary
    summary = summarization_chain.run(transcript=transcript_text)
    
    return f"Successfully processed video '{video_id}'.", summary, ""

def answer_question(question: str):
    """Answers a question based on the processed video's context."""
    if not question:
        return "Please enter a question."
    if not video_data_store["vector_store"]:
        return "Please process a video first before asking a question."

    # Retrieve relevant context from the vector store
    relevant_docs = video_data_store["vector_store"].similarity_search(question, k=5)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Generate answer
    answer = qa_chain.run(context=context, question=question)
    return answer

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# YouTube AI Video Assistant")
    gr.Markdown("Enter a YouTube URL to get a summary and ask questions about its content.")

    with gr.Row():
        video_url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...", scale=4)
        process_btn = gr.Button("Get Summary", variant="primary", scale=1)

    status_output = gr.Textbox(label="Status", interactive=False)
    summary_output = gr.Textbox(label="Video Summary", lines=8, interactive=False)
    
    with gr.Group():
        question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="What is the main topic of the video?")
        ask_btn = gr.Button("Ask Question")
        answer_output = gr.Textbox(label="Answer", lines=5, interactive=False)

    process_btn.click(
        fn=process_video_and_get_summary,
        inputs=[video_url_input],
        outputs=[status_output, summary_output, answer_output] # Clear previous answer
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output]
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)