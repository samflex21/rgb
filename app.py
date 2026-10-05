import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. Find the PDF
# --------------------------------------------------

PDF_FOLDER = "document"

pdf_files = [
    file for file in os.listdir(PDF_FOLDER)
    if file.lower().endswith(".pdf")
]

if not pdf_files:
    raise FileNotFoundError(
        "No PDF found. Put a PDF inside the 'documents' folder."
    )

pdf_path = os.path.join(PDF_FOLDER, pdf_files[0])

print(f"Loading PDF: {pdf_path}")


# --------------------------------------------------
# 2. Load the PDF
# --------------------------------------------------

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages.")


# --------------------------------------------------
# 3. Split the document into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# --------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# 5. Store embeddings in ChromaDB
# --------------------------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("PDF stored in ChromaDB.")


# --------------------------------------------------
# 6. Create retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# --------------------------------------------------
# 7. Connect Llama 3.2
# --------------------------------------------------

llm = OllamaLLM(
    model="llama3.2"
)


# --------------------------------------------------
# 8. Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions about a PDF document.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I cannot find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# 9. Question-answer loop
# --------------------------------------------------

print("\nRAG system is ready!")
print("Ask questions about your PDF.")
print("Type 'exit' to stop.\n")


while True:

    question = input("Question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    formatted_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(formatted_prompt)

    print("\nAnswer:")
    print(response)

    print("\nSources:")
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "Unknown")
        if isinstance(page, int):
            page += 1

        print(f"- Page {page}")

    print()