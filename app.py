from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import fitz
import cv2
from werkzeug.utils import secure_filename
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTImage
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = Flask(__name__)
CORS(app)

input_folder = "uploads"
output_folder = "uploadimages"
model_weights = 'modelweights/best.pt'
source = "uploadimages"
root_folder = "ultralytics_crop"

ALLOWED_EXTENSIONS = {'pdf'}
app.config['input_folder'] = input_folder

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_yolo_images(model_weights, source):
    model = YOLO(model_weights)
    names = model.names

    image_dir = source
    assert os.path.exists(image_dir), "Image directory not found"

    output_dir = "ultralytics_crop"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_dir, image_file)
            im0 = cv2.imread(image_path)
            if im0 is None:
                continue

            results = model.predict(im0, show=False)
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            for i, (box, cls) in enumerate(zip(boxes, clss)):
                class_name = names[int(cls)]
                class_output_dir = os.path.join(output_dir, class_name)
                if not os.path.exists(class_output_dir):
                    os.mkdir(class_output_dir)

                annotator = Annotator(im0, line_width=2, example=names)
                annotator.box_label(box, color=colors(int(cls), True), label=class_name)

                crop_obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                output_file = f"{os.path.splitext(image_file)[0]}{class_name}{i}.png"
                output_path = os.path.join(class_output_dir, output_file)
                cv2.imwrite(output_path, crop_obj)

    cv2.destroyAllWindows()

def image_base_info(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    ocr_text = ' '.join([entry[1] for entry in result])
    return ocr_text

def create_image_dict(root_folder):
    image_dict = {}
    page_numbers = {}

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                image_info = extract_info_from_filename(filename)
                if image_info['page_number'] not in page_numbers:
                    page_numbers[image_info['page_number']] = len(page_numbers) + 1

                page_number = page_numbers[image_info['page_number']]
                reference_number = f"{image_info['file_number']}-{page_number}-{len(image_dict) + 1}"
                image_path = os.path.join(folder_path, filename)

                image_key = os.path.splitext(filename)[0]
                image_dict[image_key] = {
                    'page_number': page_number,
                    'reference_number': reference_number,
                    'base_info': image_base_info(image_path),
                    'class': folder_name,
                    'path':  image_path
                }
    return image_dict

def extract_info_from_filename(filename):
    filename = os.path.splitext(filename)[0]
    filename = os.path.splitext(filename)[0].replace("-", "_")
    parts = filename.split('_')
    file_number = parts[1]
    page_number = parts[3]
    match = re.search(r'^(\d+)', page_number)
    numeric_part = None
    if match:
        numeric_part = match.group(1)
    return {
        'file_number': file_number,
        'page_number': numeric_part,
    }

def data_ingestion(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                loader = PDFMinerLoader(pdf_path, extract_images=False)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                pdf_document = fitz.open(pdf_path)
                for page_number in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_number)
                    pixmap = page.get_pixmap()
                    image_name = f"{os.path.splitext(file)[0]}_page_{page_number + 1}.jpg"
                    image_path = os.path.join(output_folder, image_name)
                    pixmap.save(image_path)
                pdf_document.close()

    get_yolo_images(model_weights, output_folder)
    image_dict = create_image_dict(root_folder)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=3000, length_function=len)
    texts = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db_text = Chroma.from_documents(texts, embeddings, persist_directory="db_text")
    db_text.persist()
    db_text = None
    return image_dict

image_dict = data_ingestion(input_folder, output_folder)

def get_image_reference(query, image_dict):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    docs = []
    for reference_no, image_info in image_dict.items():
        doc = Document(page_content=image_info["base_info"], metadata={"reference_number": reference_no})
        docs.append(doc)
    db.add_documents(docs)
    docs = db.similarity_search(query)
    if docs:
        image_key = docs[0].metadata["reference_number"]
        image_path = image_dict[image_key]['path']
        print("Information of the image:", image_dict[image_key])
        Image.open(image_path).show()
        return image_dict[image_key]
    else:
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_answer(query, image_dict=image_dict):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db_text = Chroma(embedding_function=embeddings, persist_directory="db_text") 
    docs = db_text.similarity_search(query)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    print(response)
    image_info = get_image_reference(query, image_dict)
    return response["output_text"]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['input_folder'], filename))
        file_path = os.path.join(app.config['input_folder'], filename)
        return jsonify({'file_path': file_path})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    file_path = data.get('file_path')
    question = data.get('question')

    if not file_path:
        return jsonify({'error': 'File path not provided'})
    document_path = file_path
    answer = process_answer(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=4000)
