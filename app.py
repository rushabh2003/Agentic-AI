# app.py - Updated Flask application with Hugging Face API integration
from flask import Flask, request, jsonify, render_template
import requests
import json
import os

api_key = os.environ.get("HF_API_KEY")

from typing import Dict, List, Any, Optional
import time
import re



app = Flask(__name__)

# Configuration
# Replace this with your Hugging Face API key
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")  
# Default to a Llama3 model on Hugging Face
DEFAULT_MODEL = "google/gemma-7b" 
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/"

class Agent:
    def __init__(self, name: str, role: str, system_prompt: str, model: str = DEFAULT_MODEL):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.conversation_history = []
    
    def query_huggingface(self, prompt: str) -> str:
        """Query the Hugging Face API with the agent's context"""
        full_prompt = f"{self.system_prompt}\n\nConversation history:\n"
        
        # Add conversation history (limited to last 5 exchanges for context)
        for entry in self.conversation_history[-5:]:
            full_prompt += f"{entry['role']}: {entry['content']}\n"
        
        full_prompt += f"Human: {prompt}\n{self.name}:"
        
        # Set up headers with the API key
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Payload for text generation
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": 0.7,
                "max_new_tokens": 1024,
                "return_full_text": False
            }
        }
        
        try:
            # Make request to the Hugging Face API
            response = requests.post(
                f"{HUGGINGFACE_API_URL}{self.model}",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            # The structure of the response varies by model, but most return a list
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
            
            # Handle other response formats
            return str(result)
            
        except Exception as e:
            return f"Error communicating with Hugging Face API: {str(e)}"
    
    def process_task(self, task: str) -> str:
        """Process a task and update conversation history"""
        self.conversation_history.append({"role": "Human", "content": task})
        response = self.query_huggingface(task)
        self.conversation_history.append({"role": self.name, "content": response})
        return response

class AgentSystem:
    def __init__(self):
        # Create the Researcher agent
        researcher_prompt = """You are the Research Agent, an AI specialized in gathering, analyzing, and summarizing information. 
        Your primary functions are:
        - Break down complex topics into key areas to investigate
        - Synthesize information clearly and concisely
        - Present multiple perspectives on debated topics
        - Identify reliable sources and patterns in data
        - Focus on providing factual, well-organized information
        
        Always structure your research in a clear way with sections and bullet points when appropriate.
        """
        
        # Create the Writer agent
        writer_prompt = """You are the Writer Agent, an AI specialized in crafting compelling content based on research.
        Your primary functions are:
        - Transform research into engaging narrative content
        - Adapt your writing style to different requirements (formal, conversational, etc.)
        - Structure content logically with clear introductions and conclusions
        - Use vivid language and appropriate metaphors to explain complex concepts
        - Maintain consistent tone and voice throughout the content
        
        Always focus on clarity and engagement in your writing.
        """
        
        # Create the Resume Builder agent
        resume_prompt = """You are the Resume Agent, an AI specialized in creating professional LaTeX resumes.
        Your primary functions are:
        - Create clean, professional LaTeX resume code
        - Structure resumes according to best practices
        - Optimize content for ATS systems and human readers
        - Highlight key accomplishments and skills effectively
        - Use appropriate LaTeX formatting and structure
        
        Always provide complete, compilable LaTeX code using a professional template.
        """
        
        # Create the LinkedIn Post agent
        linkedin_prompt = """You are the LinkedIn Agent, an AI specialized in crafting engaging LinkedIn posts.
        Your primary functions are:
        - Create attention-grabbing opening lines
        - Structure content for maximum engagement
        - Include appropriate hashtags and calls to action
        - Adapt tone to professional context while remaining authentic
        - Optimize post length and formatting for LinkedIn's platform
        
        Always focus on professionalism while maintaining an engaging, authentic voice.
        """
        
        # You can use different models for different agents if needed
        researcher_model = "mistralai/Mistral-7B-Instruct-v0.2"  # This may already be accessible
        writer_model = "google/gemma-7b"  # Change from Meta-Llama-3-8B
        resume_model = "mistralai/Mistral-7B-Instruct-v0.2"  # This looks fine
        linkedin_model = "google/gemma-7b"  # Change from Meta-Llama-3-8B
        
        self.researcher = Agent("Researcher", "Research Specialist", researcher_prompt, researcher_model)
        self.writer = Agent("Writer", "Content Creator", writer_prompt, writer_model)
        self.resume_builder = Agent("ResumeBuilder", "Resume Expert", resume_prompt, resume_model)
        self.linkedin_agent = Agent("LinkedInAgent", "Social Media Expert", linkedin_prompt, linkedin_model)
        self.workflow_history = []
    
    def execute_workflow(self, user_task: str) -> Dict[str, Any]:
        """Execute a full workflow with both agents"""
        workflow_id = f"task_{int(time.time())}"
        
        # Step 1: Researcher investigates the topic
        research_task = f"Research the following topic in depth: {user_task}"
        research_result = self.researcher.process_task(research_task)
        
        # Step 2: Writer creates content based on research
        writing_task = f"Create engaging content based on this research: {research_result}"
        final_content = self.writer.process_task(writing_task)
        
        # Store workflow results
        workflow_result = {
            "id": workflow_id,
            "user_task": user_task,
            "research": research_result,
            "final_content": final_content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.workflow_history.append(workflow_result)
        
        return workflow_result

    def create_resume(self, user_info: Dict[str, Any]) -> str:
        """Create a LaTeX resume based on user information"""
        # Format user information for the resume agent
        prompt = f"""Create a professional LaTeX resume for the following person:
        
        Name: {user_info.get('name', 'N/A')}
        Email: {user_info.get('email', 'N/A')}
        Phone: {user_info.get('phone', 'N/A')}
        Education: {user_info.get('education', 'N/A')}
        Experience: {user_info.get('experience', 'N/A')}
        Skills: {user_info.get('skills', 'N/A')}
        Projects: {user_info.get('projects', 'N/A')}
        
        Create a complete, compilable LaTeX resume using a clean, professional template. Include all sections (Education, Experience, Skills, Projects) with proper formatting.
        """
        
        # Process with the resume builder agent
        latex_resume = self.resume_builder.process_task(prompt)
        
        # Extract just the LaTeX code (in case the agent includes explanations)
        latex_code = self.extract_latex_code(latex_resume)
        
        return latex_code if latex_code else latex_resume
    
    def extract_latex_code(self, text: str) -> str:
        """Extract LaTeX code from text, looking for common LaTeX document patterns"""
        # Try to find the LaTeX document between \documentclass and \end{document}
        full_doc_match = re.search(r'(\\documentclass.*?\\end{document})', text, re.DOTALL)
        if full_doc_match:
            return full_doc_match.group(1)
        
        # If no complete document, look for LaTeX-like content
        latex_sections = re.findall(r'(\\begin{.*?}.*?\\end{.*?})', text, re.DOTALL)
        if latex_sections:
            return '\n\n'.join(latex_sections)
            
        return text
    
    def create_linkedin_post(self, post_info: Dict[str, Any]) -> str:
        """Create a LinkedIn post based on user information , do not give any explanation just return the post"""
        prompt = f"""Create an engaging LinkedIn post about:
        
        Topic: {post_info.get('topic', 'N/A')}
        Key points: {post_info.get('key_points', 'N/A')}
        Tone: {post_info.get('tone', 'Professional')}
        Target audience: {post_info.get('audience', 'Professionals')}
        
        Create a complete LinkedIn post with an attention-grabbing opening, meaningful content, and appropriate hashtags.
        The post should be formatted properly for LinkedIn (with paragraphs, emojis if appropriate, and hashtags).
        """
        
        # Process with the LinkedIn agent
        linkedin_post = self.linkedin_agent.process_task(prompt)
        return linkedin_post

# Create the agent system
agent_system = AgentSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resume-builder')
def resume_builder():
    return render_template('resume_builder.html')

@app.route('/linkedin-post')
def linkedin_post():
    return render_template('linkedin_post.html')

@app.route('/api/task', methods=['POST'])
def process_task():
    data = request.json
    user_task = data.get('task', '')
    
    if not user_task:
        return jsonify({"error": "No task provided"}), 400
    
    # Execute the agent workflow
    try:
        result = agent_system.execute_workflow(user_task)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/create-resume', methods=['POST'])
def create_resume():
    data = request.json
    
    if not data:
        return jsonify({"error": "No resume information provided"}), 400
    
    try:
        latex_resume = agent_system.create_resume(data)
        return jsonify({"latex_code": latex_resume})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/create-linkedin-post', methods=['POST'])
def create_linkedin_post():
    data = request.json
    
    if not data:
        return jsonify({"error": "No post information provided"}), 400
    
    try:
        linkedin_post = agent_system.create_linkedin_post(data)
        return jsonify({"post_content": linkedin_post})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify({"history": agent_system.workflow_history})

if __name__ == '__main__':
    app.run(debug=True, port=5000)