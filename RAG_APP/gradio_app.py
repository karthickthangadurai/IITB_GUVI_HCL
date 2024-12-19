import gradio as gr
from pathlib import Path
from typing import List, Tuple

# Import the RAG functions from the simplified implementation
from rag import initialize_models, process_pdf, get_answer

def create_ui():
    """Create and launch the Gradio interface."""
    # Initialize models at startup
    llm, embeddings = initialize_models()
    
    # State variables (using function closure instead of class attributes)
    current_pdf = None
    vector_store = None
    
    def load_pdf(file_obj) -> str:
        """Load a PDF file into the RAG system."""
        nonlocal current_pdf, vector_store
        
        if file_obj is None:
            return "No file selected."
        
        try:
            vector_store = process_pdf(file_obj.name, embeddings)
            current_pdf = Path(file_obj.name).name
            return f"Successfully loaded PDF: {current_pdf}"
        except Exception as e:
            return f"Error loading PDF: {str(e)}"

    def chat(
        message: str, 
        history: List[Tuple[str, str]], 
        use_pdf: bool
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Handle chat interactions with or without PDF context."""
        if not message.strip():
            return "", history

        try:
            # Get answer using appropriate context
            answer = get_answer(
                question=message,
                llm=llm,
                vector_store=vector_store if use_pdf and current_pdf else None
            )
            
            # Add context information if using PDF
            if use_pdf and current_pdf is not None:
                answer = f"[Using PDF: {current_pdf}]\n\n{answer}"
            else:
                answer = "[Using general knowledge]\n\n" + answer

            # Update history
            history.append((message, answer))
            return "", history
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append((message, error_msg))
            return "", history

    def clear_chat() -> List[Tuple[str, str]]:
        """Clear the chat history."""
        return []

    # Define the interface
    with gr.Blocks(title="PDF Chat System") as interface:
        gr.Markdown("# PDF Chat System")
        gr.Markdown("Upload a PDF to chat about its contents, or chat without PDF context for general knowledge.")
        
        with gr.Row():
            # Left sidebar for PDF upload and controls
            with gr.Column(scale=1):
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"]
                )
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False
                )
                use_pdf = gr.Checkbox(
                    label="Use PDF Context",
                    value=True,
                    info="When checked, answers will be based on the uploaded PDF"
                )
                clear_btn = gr.Button("Clear Chat")

            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=600,
                    show_label=False
                )
                message = gr.Textbox(
                    placeholder="Type your message here...",
                    container=False,
                    scale=7
                )
                submit_btn = gr.Button("Send")

        # Set up event handlers
        pdf_upload.change(
            fn=load_pdf,
            inputs=[pdf_upload],
            outputs=[upload_status]
        )
        
        submit_btn.click(
            fn=chat,
            inputs=[message, chatbot, use_pdf],
            outputs=[message, chatbot]
        )
        
        message.submit(
            fn=chat,
            inputs=[message, chatbot, use_pdf],
            outputs=[message, chatbot]
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot]
        )

    # Launch the interface
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )

if __name__ == "__main__":
    create_ui()