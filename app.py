import os
import datetime
import gradio as gr
from markdown_pdf import MarkdownPdf, Section
from resume_crew.crew import ResumeCrew

# Set backend directories for Hugging Face Spaces
UPLOAD_DIR = "/tmp/uploads"
OUTPUT_DIR = "/tmp/output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_md_to_pdf(md_path: str) -> str:
    """Convert Markdown to PDF and save it in OUTPUT_DIR."""
    if not os.path.isfile(md_path):
        return ""
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    pdf_obj = MarkdownPdf(toc_level=2)
    pdf_obj.add_section(Section(md_content))

    pdf_filename = os.path.splitext(os.path.basename(md_path))[0] + ".pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    
    pdf_obj.save(pdf_path)
    return pdf_path if os.path.isfile(pdf_path) else ""

def process_resume(model_choice, new_resume, company_name, job_url, openai_api_key, serper_api_key):
    """Handles file upload, API keys, resume processing, and returns PDFs."""
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    if not openai_api_key or not serper_api_key:
        return ("Error: Please enter both OpenAI and Serper API keys.", None, None, None)

    if new_resume is None or not hasattr(new_resume, "name") or not new_resume.name.strip():
        return ("Error: Please upload a resume.", None, None, None)

    resume_path = os.path.join(UPLOAD_DIR, new_resume.name)
    
    with open(resume_path, "wb") as f:
        f.write(new_resume.read())

    crew_instance = ResumeCrew(
        model=model_choice,
        openai_api_key=openai_api_key,  
        serper_api_key=serper_api_key,  
        resume_pdf_path=resume_path  
    )

    crew = crew_instance.crew()
    if crew:
        crew.kickoff(inputs={'job_url': job_url, 'company_name': company_name})

    def md_to_pdf_in_dir(md_filename):
        md_path = os.path.join(OUTPUT_DIR, md_filename)
        if os.path.isfile(md_path):
            return convert_md_to_pdf(md_path)
        return ""

    pdf_opt = md_to_pdf_in_dir("optimized_resume.md")
    pdf_final = md_to_pdf_in_dir("final_report.md")
    pdf_int = md_to_pdf_in_dir("interview_questions.md")

    message = f"Processing completed using model {model_choice}. Files are stored temporarily."

    return message, pdf_opt, pdf_final, pdf_int

model_choices = {
    "GPT-4o-mini": "gpt-4o-mini-2024-07-18",
    "GPT-4o": "gpt-4o-2024-08-06",
    "o3-mini": "o3-mini-2025-01-31",
    "o1-mini": "o1-mini-2024-09-12"
}

with gr.Blocks(css=".output-column { width: 700px; }") as demo:
    with gr.Row():
        with gr.Column(scale=0.8):
            gr.Markdown("## Resume Optimization System")
            openai_api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
            serper_api_key_input = gr.Textbox(label="Serper API Key", type="password")
            model_dropdown = gr.Dropdown(choices=list(model_choices.values()), label="Select Model")
            new_resume_file = gr.File(label="Upload New Resume PDF", file_types=[".pdf"])
            company_name_text = gr.Textbox(label="Company Name")
            job_url_text = gr.Textbox(label="Job URL")
            run_button = gr.Button("Run")

    # Define output components
    result_text = gr.Textbox(label="Status", interactive=False)
    pdf_opt_out = gr.File(label="Optimized Resume PDF")
    pdf_final_out = gr.File(label="Final Report PDF")
    pdf_int_out = gr.File(label="Interview Questions PDF")

    # Connect button
    run_button.click(
        process_resume, 
        inputs=[model_dropdown, new_resume_file, company_name_text, job_url_text, openai_api_key_input, serper_api_key_input],
        outputs=[result_text, pdf_opt_out, pdf_final_out, pdf_int_out]
    )

demo.launch()
