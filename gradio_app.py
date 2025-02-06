import os
import shutil
import datetime
import json
import gradio as gr

# 1. Import markdown-pdf to handle MD -> PDF conversion
from markdown_pdf import MarkdownPdf, Section

# 2. Import PDF from gradio-pdf for inline PDF viewing
from gradio_pdf import PDF

from resume_crew.crew import ResumeCrew
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource


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


def process_resume(model_choice, new_resume, company_name, job_url):
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # --- Ensure a resume file is uploaded ---
    if new_resume is None or not (hasattr(new_resume, "name") and new_resume.name.strip() != ""):
        # Return 7 outputs, but only the first (status) is meaningful here
        return ("Error: Please upload a resume.", None, None, None, None, None, None)
    
    # --- Save the uploaded file ---
    if hasattr(new_resume, "read"):
        original_filename = os.path.basename(new_resume.name)
        file_data = new_resume.read()
    else:
        original_filename = os.path.basename(new_resume)
        file_data = None
    base_filename, ext = os.path.splitext(original_filename)
    new_resume_filename = f"{base_filename}_{current_date}{ext}"
    knowledge_path = os.path.join("knowledge", new_resume_filename)
    
    if file_data is not None:
        with open(knowledge_path, "wb") as f:
            f.write(file_data)
    else:
        shutil.copy(new_resume, knowledge_path)
    
    used_resume_filename = new_resume_filename
    
    # --- Instantiate ResumeCrew with the selected model ---
    crew_instance = ResumeCrew(model=model_choice)
    crew_instance.resume_pdf = PDFKnowledgeSource(file_paths=[used_resume_filename])
    
    # Run the process
    crew_instance.crew().kickoff(inputs={'job_url': job_url, 'company_name': company_name})
    
    # --- Retrieve output files ---
    job_analysis_path = os.path.join("output", "job_analysis.json")
    try:
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
    folder_name = f"{company_name}_{position_name}_{candidate_name}_{current_date}"
    new_output_dir = os.path.join("output", folder_name)
    os.makedirs(new_output_dir, exist_ok=True)
    
    # --- Move generated output files ---
    for filename in os.listdir("output"):
        file_path = os.path.join("output", filename)
        if file_path == new_output_dir:
            continue
        if filename.endswith(".json") or filename.endswith(".md"):
            if os.path.isfile(file_path):
                shutil.move(file_path, os.path.join(new_output_dir, filename))
    
    # --- Convert each MD to PDF ---
    def md_to_pdf_in_dir(md_filename):
        md_path = os.path.join(new_output_dir, md_filename)
        if os.path.isfile(md_path):
            return convert_md_to_pdf(md_path)
        return ""
    
    pdf_opt = md_to_pdf_in_dir("optimized_resume.md")
    pdf_final = md_to_pdf_in_dir("final_report.md")
    pdf_int = md_to_pdf_in_dir("interview_questions.md")
    
    message = f"Processing completed using model {model_choice}. Output saved in: {new_output_dir}"
    
    # RETURN EXACTLY 7 outputs to match your UI
    return (
        message,      # 1) Goes to status_output (Textbox)
        pdf_opt,      # 2) PDF viewer for optimized resume
        pdf_opt,      # 3) File for downloading optimized resume
        pdf_final,    # 4) PDF viewer for final report
        pdf_final,    # 5) File for downloading final report
        pdf_int,      # 6) PDF viewer for interview questions
        pdf_int       # 7) File for downloading interview questions
    )


# --- Define model options ---
model_choices = {
    "GPT-4o-mini": "gpt-4o-mini-2024-07-18",
    "GPT-4o": "gpt-4o-2024-08-06",
    "o3-mini": "o3-mini-2025-01-31",
    "o1-mini": "o1-mini-2024-09-12"
}


with gr.Blocks(css=".output-column { width: 700px; }") as demo:
    with gr.Row():
        with gr.Column(scale=0.8):  # ⬅️ Narrower input section
            gr.Markdown("## Resume Optimization System")
            model_dropdown = gr.Dropdown(
                choices=list(model_choices.values()),
                label="Select Model",
                value="o3-mini-2025-01-31",
                interactive=True,
                info="Select the model to use for processing."
            )
            new_resume_file = gr.File(label="Upload New Resume PDF", file_types=[".pdf"])
            company_name_text = gr.Textbox(label="Company Name", placeholder="Enter company name")
            job_url_text = gr.Textbox(label="Job URL", placeholder="Enter job posting URL")
            run_button = gr.Button("Run")

        with gr.Column(scale=1.5, elem_classes="output-column"):  # ⬅️ Larger output section
            gr.Markdown("## Processing Status")
            status_output = gr.Textbox(label="Status")

            with gr.Tabs():
                with gr.Tab("Optimized Resume PDF"):
                    pdf_opt_download = gr.File(label="Download Optimized Resume")  # ⬅️ Download button above
                    pdf_opt_viewer = PDF(label="View Optimized Resume")  # ⬅️ Viewer below

                with gr.Tab("Final Report PDF"):
                    pdf_final_download = gr.File(label="Download Final Report")
                    pdf_final_viewer = PDF(label="View Final Report")

                with gr.Tab("Interview Questions PDF"):
                    pdf_int_download = gr.File(label="Download Interview Questions")
                    pdf_int_viewer = PDF(label="View Interview Questions")

    run_button.click(
        process_resume,
        inputs=[model_dropdown, new_resume_file, company_name_text, job_url_text],
        outputs=[
            status_output,
            pdf_opt_viewer, pdf_opt_download,
            pdf_final_viewer, pdf_final_download,
            pdf_int_viewer, pdf_int_download
        ]
    )

if __name__ == "__main__":
    demo.launch()
