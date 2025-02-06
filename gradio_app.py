import gradio as gr
import os
import datetime
import shutil
import json
from resume_crew.crew import ResumeCrew
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

def process_resume(model_choice, new_resume, company_name, job_url):
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # --- Ensure a resume file is uploaded ---
    if new_resume is None or not (hasattr(new_resume, "name") and new_resume.name.strip() != ""):
        return ("Error: Please upload a resume.", "", "", "")
    
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
    
    used_resume_filename = new_resume_filename  # Always use the newly uploaded resume
    
    # --- Prepare inputs for the Crew process ---
    inputs = {
        'job_url': job_url,
        'company_name': company_name
    }
    
    # Instantiate ResumeCrew and update attributes.
    crew_instance = ResumeCrew()
    crew_instance.resume_pdf = PDFKnowledgeSource(file_paths=used_resume_filename)
    crew_instance.model = model_choice
    
    # Run the Crew process (this may take a while).
    crew_instance.crew().kickoff(inputs=inputs)
    
    # --- Retrieve the role (job title) from job_analysis.json ---
    job_analysis_path = os.path.join("output", "job_analysis.json")
    try:
        with open(job_analysis_path, "r") as f:
            job_data = json.load(f)
        position_name = job_data.get("job_title", "position")
    except Exception as e:
        position_name = "position"
    
    # --- Retrieve the candidate name from optimized_resume.md ---
    optimized_resume_path = os.path.join("output", "optimized_resume.md")
    candidate_name = "candidate"
    try:
        with open(optimized_resume_path, "r") as f:
            first_line = f.readline()
            if first_line.startswith("#"):
                candidate_name = first_line.lstrip("#").strip().replace(" ", "_")
    except Exception as e:
        candidate_name = "candidate"
    
    # --- Create the output folder directly inside "output" ---
    folder_name = f"{company_name}_{position_name}_{candidate_name}_{current_date}"
    new_output_dir = os.path.join("output", folder_name)
    os.makedirs(new_output_dir, exist_ok=True)
    
    # --- Move generated output files (*.json and *.md) into the new folder ---
    for filename in os.listdir("output"):
        file_path = os.path.join("output", filename)
        # Skip the output folder we just created.
        if file_path == new_output_dir:
            continue
        if filename.endswith(".json") or filename.endswith(".md"):
            if os.path.isfile(file_path):
                shutil.move(file_path, os.path.join(new_output_dir, filename))
    
    # --- Read the markdown contents from the new folder ---
    def read_md(file_name):
        try:
            with open(os.path.join(new_output_dir, file_name), "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading {file_name}: {str(e)}"
    
    md_opt = read_md("optimized_resume.md")
    md_final = read_md("final_report.md")
    md_int = read_md("interview_questions.md")
    
    message = f"Processing completed. Output saved in: {new_output_dir}"
    return (message, md_opt, md_final, md_int)

# --- Define model options ---
model_choices = {
    "GPT-4o-mini": "gpt-4o-mini-2024-07-18",
    "GPT-4o": "gpt-4o-2024-08-06",
    "o3-mini": "o3-mini-2025-01-31",
    "o1-mini": "o1-mini-2024-09-12"
}

# --- Build the UI using Gradio Blocks ---
# Now the left pane (inputs) will be narrower (scale=1) and the right pane (outputs) wider (scale=2).
with gr.Blocks(css=".output-column { width: 600px; }") as demo:
    with gr.Row():
        with gr.Column(scale=1):
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
        with gr.Column(scale=2, elem_classes="output-column"):
            gr.Markdown("## Processing Status")
            status_output = gr.Textbox(label="Status")
            with gr.Tabs():
                with gr.Tab("Optimized Resume"):
                    md_opt_output = gr.Markdown(label="Optimized Resume")
                with gr.Tab("Final Report"):
                    md_final_output = gr.Markdown(label="Final Report")
                with gr.Tab("Interview Questions"):
                    md_int_output = gr.Markdown(label="Interview Questions")
                    
    run_button.click(
        process_resume,
        inputs=[model_dropdown, new_resume_file, company_name_text, job_url_text],
        outputs=[status_output, md_opt_output, md_final_output, md_int_output]
    )

if __name__ == "__main__":
    demo.launch()
