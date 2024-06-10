# DocBot-Chatbot-based-document-assistance-system-

- [About](#about)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Example Usage](#example-usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## About
This project is a PDF Question Answering System built using Flask, YOLO, EasyOCR, HuggingFace, and LangChain libraries. The system allows users to upload PDF files, extract text and images, and ask questions about the content. The system responds with detailed answers based on the extracted data.

## Features
- Extracts text and images from PDF files.
- Converts PDF pages to images for enhanced image detection.
- Uses YOLO for detecting images, arrows, numbers, and other annotations from the converted PDF images.
- Utilizes EasyOCR for text extraction from detected images.
- Provides a question-answering interface powered by LangChain and HuggingFace models.
- Integrates Google Generative AI for enhanced question answering.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pdf-question-answering.git
   cd pdf-question-answering

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt

4. **Set up environment variables**:
   Create a '.env' file in the project root directory.
   Add your Google API key to the '.env' file:
   ```makefile
   GOOGLE_API_KEY=your-google-api-key

## Directory Structure
- 'uploads' : Directory to store uploaded PDF files.
- 'uploadimages' : Directory to store images extracted from PDFs.
- 'modelweights' : Directory to store YOLO model weights.
- 'db_text' : Directory to store Chroma vector store data.

## Usage

### Running the Application

- Ensure your virtual environment is activated.
- Start the Flask application:
  ```bash
   python app.py
- The application will run on 'http://127.0.0.1:4000' .

## Example Usage

- Upload a PDF file using the /upload endpoint.
- Use the /ask_question endpoint to ask questions about the uploaded PDF file.

## Dependencies

- Flask
- Flask-CORS
- langchain-community
- pdfminer
- PyPDF2
- pdfplumber
- pytesseract
- transformers
- torch
- easyocr
- ultralytics
- pdf2image
- dotenv

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [HuggingFace](https://huggingface.co)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [YOLO](https://github.com/ultralytics/ultralytics)
- [Flask](https://github.com/pallets/flask)

## Creators

- [Yash Shrivastava]()
