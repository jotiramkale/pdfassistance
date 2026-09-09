

# ==========================================
# IMPORTS
# ==========================================

import os

from flask import Flask, render_template, request

from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq


# ==========================================
# CONFIG
# ==========================================

GROQ_API_KEY="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"

MODEL_NAME="openai/gpt-oss-120b"

UPLOAD_FOLDER="uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# GROQ MODEL
# ==========================================

llm = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY
)


# ==========================================
# EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# CHROMADB
# ==========================================

chroma_client = chromadb.Client()

collection = None


# ==========================================
# READ PDF
# ==========================================

def read_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


# ==========================================
# SPLIT TEXT
# ==========================================

def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=200
    )

    return splitter.split_text(text)


# ==========================================
# STORE PDF IN CHROMADB
# ==========================================

def create_vector_database(chunks):

    global collection

    try:
        chroma_client.delete_collection("pdf_assistant")
    except:
        pass

    collection = chroma_client.create_collection(
        name="pdf_assistant"
    )

    embeddings = embedding_model.encode(chunks)

    for i, chunk in enumerate(chunks):

        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embeddings[i].tolist()]
        )


# ==========================================
# RETRIEVE CONTEXT
# ==========================================

def retrieve_context(question):

    question_embedding = embedding_model.encode(
        question
    )

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=3
    )

    context = "\n\n".join(
        results["documents"][0]
    )

    return context


# ==========================================
# PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful PDF Assistant.

Answer ONLY using the context below.

If answer is not available say:

I could not find that information in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = prompt | llm | StrOutputParser()


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    message = ""

    if request.method == "POST":

        pdf_file = request.files["pdf"]

        if pdf_file:

            pdf_path = os.path.join(
                UPLOAD_FOLDER,
                pdf_file.filename
            )

            pdf_file.save(pdf_path)

            text = read_pdf(pdf_path)

            chunks = split_text(text)

            create_vector_database(chunks)

            message = "PDF Uploaded Successfully"

    return render_template(
        "index.html",
        message=message,
        answer=""
    )


# ==========================================
# ASK QUESTION
# ==========================================

@app.route("/ask", methods=["POST"])
def ask():

    question = request.form["question"]

    context = retrieve_context(question)

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return render_template(
        "index.html",
        answer=answer,
        message=""
    )


# ==========================================
# RUN
# ==========================================

def main():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()