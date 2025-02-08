import os
import sys
import shutil
import datetime
import json
import gradio as gr

# Ensure `src` is in Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from markdown_pdf import MarkdownPdf, Section
from gradio_pdf import PDF
from resume_crew.crew import ResumeCrew
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

# Set backend directories for Hugging Face Spaces
UPLOAD_DIR = "/tmp/uploads"
OUTPUT_DIR = "/tmp/output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_md_to_pdf(md_path: str) -> str:
    """
    Convert a local .md file to .pdf using markdown-pdf.
    Returns the resulting PDF file path, or an empty string if conversion fails.
    """
    if not os.path.isfile(md_path):
        return ""
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    pdf_obj = MarkdownPdf(toc_level=2)
    pdf_obj.add_section(Section(md_content))
    
    pdf_path = os.path.splitext(md_path)[0] + ".pdf"
    pdf_obj.save(pdf_path)
    return pdf_path if os.path.isfile(pdf_path) else ""

def process_resume(openai_api_key, serper_api_key, model_choice, new_resume, company_name, job_url):
    """
    Processes the uploaded resume using ResumeCrew and converts the output Markdown files to PDFs.
    Handles errors gracefully and stops execution upon failure.
    """
    try:
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        
        # --- Ensure a resume file is uploaded ---
        if new_resume is None or not (hasattr(new_resume, "name") and new_resume.name.strip() != ""):
            return ("Error: Please upload a resume.", None, None, None, None, None, None)
        
        # --- Set API keys ---
        os.environ["OPENAI_API_KEY"] = openai_api_key or ""
        os.environ["SERPER_API_KEY"] = serper_api_key or ""

        # --- Save uploaded file ---
        try:
            if hasattr(new_resume, "read"):
                original_filename = os.path.basename(new_resume.name)
                file_data = new_resume.read()
            else:
                original_filename = os.path.basename(new_resume)
                file_data = None

            base_filename, ext = os.path.splitext(original_filename)
            new_resume_filename = f"{base_filename}_{current_date}{ext}"
            physical_path = os.path.join("knowledge", new_resume_filename)
            os.makedirs("knowledge", exist_ok=True)

            if file_data is not None:
                with open(physical_path, "wb") as f:
                    f.write(file_data)
            else:
                shutil.copy(new_resume, physical_path)
        except Exception as e:
            return (f"Error saving the uploaded resume: {str(e)}", None, None, None, None, None, None)

        # --- Initialize ResumeCrew ---
        try:
            crew_instance = ResumeCrew(
                model=model_choice,
                openai_api_key=openai_api_key,
                serper_api_key=serper_api_key,
                resume_pdf_path=new_resume_filename
            )
        except Exception as e:
            return (f"Error initializing ResumeCrew: {str(e)}", None, None, None, None, None, None)

        # --- Run the resume processing ---
        try:
            crew_instance.crew().kickoff(inputs={'job_url': job_url, 'company_name': company_name})
        except Exception as e:
            return (f"Error during resume processing: {str(e)}", None, None, None, None, None, None)

        # --- Retrieve output files ---
        try:
            job_analysis_path = os.path.join("output", "job_analysis.json")
            with open(job_analysis_path, "r") as f:
                job_data = json.load(f)
            position_name = job_data.get("job_title", "position")
        except Exception:
            position_name = "position"

        optimized_resume_path = os.path.join("output", "optimized_resume.md")
        candidate_name = "candidate"
        try:
            with open(optimized_resume_path, "r") as f:
                first_line = f.readline()
                if first_line.startswith("#"):
                    candidate_name = first_line.lstrip("#").strip().replace(" ", "_")
        except Exception:
            candidate_name = "candidate"

        # --- Create the output folder ---
        try:
            folder_name = f"{company_name}_{position_name}_{candidate_name}_{current_date}"
            new_output_dir = os.path.join("output", folder_name)
            os.makedirs(new_output_dir, exist_ok=True)

            for filename in os.listdir("output"):
                file_path = os.path.join("output", filename)
                if file_path == new_output_dir:
                    continue
                if filename.endswith(".json") or filename.endswith(".md"):
                    if os.path.isfile(file_path):
                        shutil.move(file_path, os.path.join(new_output_dir, filename))
        except Exception as e:
            return (f"Error organizing output files: {str(e)}", None, None, None, None, None, None)

        # --- Convert Markdown to PDF ---
        def md_to_pdf_in_dir(md_filename):
            try:
                md_path = os.path.join(new_output_dir, md_filename)
                if os.path.isfile(md_path):
                    return convert_md_to_pdf(md_path)
                return ""
            except Exception as e:
                return f"Error converting {md_filename} to PDF: {str(e)}"

        pdf_opt = md_to_pdf_in_dir("optimized_resume.md")
        pdf_final = md_to_pdf_in_dir("final_report.md")
        pdf_int = md_to_pdf_in_dir("interview_questions.md")

        message = f"Processing completed using model {model_choice}. Output saved in: {new_output_dir}"

        return (message, pdf_opt, pdf_opt, pdf_final, pdf_final, pdf_int, pdf_int)

    except Exception as e:
        return (f"Unexpected error: {str(e)}", None, None, None, None, None, None)

# --- Define available models ---
model_choices = {
    "GPT-4o-mini": "gpt-4o-mini-2024-07-18",
    "GPT-4o": "gpt-4o-2024-08-06",
    "o3-mini": "o3-mini-2025-01-31",
    "o1-mini": "o1-mini-2024-09-12"
}

with gr.Blocks(css=".output-column { width: 700px; }") as demo:
    with gr.Row():
        # Left pane: Input fields
        with gr.Column(scale=1):
            gr.Markdown("## Resume Optimization System")
            gr.Markdown(
                "Create an optimized resume, job research report, and interview question sheet "
                "by simply uploading your resume, entering the company name, and providing the job posting URL. "
                "This tool leverages multi-agentic AI and web search to analyze job descriptions, research the company, and "
                "tailor your resume for better ATS compatibility and job relevance."
            )
            openai_api_key_input = gr.Textbox(label="OpenAI API Key", type="password", placeholder="Enter OpenAI API Key")
            serper_api_key_input = gr.Textbox(label="Serper API Key", type="password", placeholder="Enter Serper API Key")
            model_dropdown = gr.Dropdown(
                choices=list(model_choices.values()),
                label="Select Model",
                value="gpt-4o-2024-08-06",
                interactive=True,
                info="Select the model to use for processing."
            )
            new_resume_file = gr.File(label="Upload New Resume PDF", file_types=[".pdf"])
            company_name_text = gr.Textbox(label="Company Name", placeholder="Enter company name")
            job_url_text = gr.Textbox(label="Job URL", placeholder="Enter job posting URL")
            run_button = gr.Button("Run")

        
        # Right pane: Output display
        with gr.Column(scale=2, elem_classes="output-column"):  # Scale set to an integer to avoid warnings
            gr.Markdown("## Processing Status")
            status_output = gr.Textbox(label="Status")
            with gr.Tabs():
                with gr.Tab("Optimized Resume PDF"):
                    pdf_opt_download = gr.File(label="Download Optimized Resume")
                    pdf_opt_viewer = PDF(label="View Optimized Resume")
                with gr.Tab("Final Report PDF"):
                    pdf_final_download = gr.File(label="Download Final Report")
                    pdf_final_viewer = PDF(label="View Final Report")
                with gr.Tab("Interview Questions PDF"):
                    pdf_int_download = gr.File(label="Download Interview Questions")
                    pdf_int_viewer = PDF(label="View Interview Questions")
    
    run_button.click(
        process_resume,
        inputs=[
            openai_api_key_input,
            serper_api_key_input,
            model_dropdown,
            new_resume_file,
            company_name_text,
            job_url_text
        ],
        outputs=[
            status_output,
            pdf_opt_viewer, pdf_opt_download,
            pdf_final_viewer, pdf_final_download,
            pdf_int_viewer, pdf_int_download
        ]
    )

if __name__ == "__main__":
    demo.launch()
