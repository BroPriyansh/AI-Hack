from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements=f.read().splitlines()
    
    setup(
        name="AI-Agent-Document-QA",
        version="0.0.1",
        author="Wesley Gonzales",
        description="Smart AI Agent for Document Q&A with Summarization and Source Highlighting",
        packages=find_packages(),
        install_requires=requirements,
        python_requires=">=3.10"
    )

# pip install -e .