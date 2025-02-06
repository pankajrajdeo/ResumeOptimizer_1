from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from .models import (
    JobRequirements,
    ResumeOptimization,
    CompanyResearch,
    InterviewQuestions
)

llm_base = "gpt-4o-mini-2024-07-18"
llm_advanced = "o3-mini-2025-01-31"

@CrewBase
class ResumeCrew():
    """ResumeCrew for resume optimization and interview preparation"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        """Sample resume PDF for testing from https://www.hbs.edu/doctoral/Documents/job-market/CV_Mohan.pdf"""
        self.resume_pdf = PDFKnowledgeSource(file_paths="CV_Mohan.pdf")

    @agent
    def resume_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['resume_analyzer'],
            verbose=True,
            llm=LLM(llm_advanced),
            knowledge_sources=[self.resume_pdf]
        )
    
    @agent
    def job_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['job_analyzer'],
            verbose=True,
            tools=[ScrapeWebsiteTool()],
            llm=LLM(llm_advanced)
        )

    @agent
    def company_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['company_researcher'],
            verbose=True,
            tools=[SerperDevTool()],
            llm=LLM(llm_base),
            knowledge_sources=[self.resume_pdf]
        )

    @agent
    def resume_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['resume_writer'],
            verbose=True,
            llm=LLM(llm_base)
        )

    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['report_generator'],
            verbose=True,
            llm=LLM(llm_base)
        )
    
    @agent
    def interview_question_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['interview_question_generator'],
            verbose=True,
            llm=LLM(llm_advanced)
        )


    @task
    def analyze_job_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_job_task'],
            output_file='output/job_analysis.json',
            output_pydantic=JobRequirements
        )

    @task
    def optimize_resume_task(self) -> Task:
        return Task(
            config=self.tasks_config['optimize_resume_task'],
            output_file='output/resume_optimization.json',
            output_pydantic=ResumeOptimization
        )

    @task
    def research_company_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_company_task'],
            output_file='output/company_research.json',  
            output_pydantic=CompanyResearch
        )

    @task
    def generate_resume_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_resume_task'],
            output_file='output/optimized_resume.md'
        )

    @task
    def generate_report_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_report_task'],
            output_file='output/final_report.md'
        )

    @task
    def generate_interview_questions_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_interview_questions_task'],
            output_file='output/interview_questions.json',
            output_pydantic=InterviewQuestions
        )

    @task
    def generate_interview_questions_md_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_interview_questions_md_task'],
            output_file='output/interview_questions.md'
        )


    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            process=Process.sequential,
            knowledge_sources=[self.resume_pdf]
        )
