from setuptools import setup, find_packages

setup(
    name="llm_culture",
    version="0.1.0",
    description="A project to simulate and analyze cultural evolution in populations of LLMs.",
    author="Jérémy Pérez, Corentin Léger",
    author_email="corentin.lger@gmail.com",
    url="https://github.com/flowersteam/LLM-Culture",
    packages=find_packages(),
    install_requires=[
        # in requirements.txt
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)