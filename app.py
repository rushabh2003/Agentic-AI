# app.py - Updated Flask application with additional features
from flask import Flask, request, jsonify, render_template
import requests
import json
import os
from typing import Dict, List, Any, Optional
import time
import re

app = Flask(__name__)

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"

class Agent:
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.conversation_history = []
    
    def query_ollama(self, prompt: str) -> str:
        """Query the local Ollama model with the agent's context"""
        full_prompt = f"{self.system_prompt}\n\nConversation history:\n"
        
        # Add conversation history (limited to last 5 exchanges for context)
        for entry in self.conversation_history[-5:]:
            full_prompt += f"{entry['role']}: {entry['content']}\n"
        
        full_prompt += f"Human: {prompt}\n{self.name}:"
        
        payload = {
            "model": "llama3.2:latest",
            "prompt": full_prompt,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["response"]
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    def process_task(self, task: str) -> str:
        """Process a task and update conversation history"""
        self.conversation_history.append({"role": "Human", "content": task})
        response = self.query_ollama(task)
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
        
        self.researcher = Agent("Researcher", "Research Specialist", researcher_prompt)
        self.writer = Agent("Writer", "Content Creator", writer_prompt)
        self.resume_builder = Agent("ResumeBuilder", "Resume Expert", resume_prompt)
        self.linkedin_agent = Agent("LinkedInAgent", "Social Media Expert", linkedin_prompt)
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
        """Create a LinkedIn post based on user information"""
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