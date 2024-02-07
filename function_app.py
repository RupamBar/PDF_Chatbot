import os
import sys
import re
import json
import openai

from flask import Flask, render_template, request
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

from PyPDF2 import PdfReader
from io import BytesIO
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain import PromptTemplate

app = Flask(__name__)

@app.route('/')
def hello():
    response = json.dumps({
        "message": "Hello world"
    })

    return response

# Chatbot API testing
@app.route('/demoapi', methods=['POST'])
def demoAPI():
    try:
        payload = request.get_json()
        query = payload["query"]

        ## Chunks division ##
        docFolder = "docs"
        userFolder = "User1"
        pdfFolder = "AI"

        def parse_pdf(file: BytesIO) -> List[str]:
            pdf = PdfReader(file)
            #file_name = os.path.basename(file)
            output = []
            for page in pdf.pages:
                text = page.extract_text()
                # Merge hyphenated words
                text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                # Fix newlines in the middle of sentences
                text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                # Remove multiple newlines
                text = re.sub(r"\n\s*\n", "\n\n", text)
                output.append(text)
            return output
        
        def text_to_docs(text, file_name):
            """Converts a string or list of strings to a list of Documents
            with metadata."""
            if isinstance(text, str):
                # Take a single string as one page
                text = [text]
            page_docs = [Document(page_content=page) for page in text]

            # Add page numbers as metadata
            for i, doc in enumerate(page_docs):
                doc.metadata["page"] = i + 1

            # Split pages into chunks
            doc_chunks = []

            for doc in page_docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=750,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                    chunk_overlap=0,
                )
                chunks = text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
                    )
                    # Add sources a metadata
                    doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
                    doc.metadata["file_name"] = file_name
                    doc_chunks.append(doc)
            return doc_chunks

        # Create a folder in the current working directory
        # folder_path = os.path.join(os.getcwd(), docFolder, userFolder, pdfFolder)
        folder_path = os.path.join(".\\", docFolder, userFolder, pdfFolder)
        print(folder_path, "folder_path")

        # Check if the folder already exists, and create it if not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{pdfFolder}' created successfully at: {folder_path}")
        else:
            print(f"Folder '{pdfFolder}' already exists at: {folder_path}")

        global text_chunked
        text_chunked = []

        if not os.path.exists(folder_path + "\\" + "dbs" + "\\" + "index"):
            print("Chunk conversion started")
            pdf_files = [file for file in os.listdir("./" + docFolder + "/" + userFolder + "/" + pdfFolder) if file.endswith('.pdf')]
            # print(pdf_files, "pdf_files")

            for pdf_file in pdf_files:
                with open("./" + docFolder + "/" + userFolder + "/" + pdfFolder + "/" + pdf_file, 'rb') as f:
                    doc = parse_pdf(f)
                    file_name = os.path.splitext(docFolder)[0]
                    text_chunked += text_to_docs(doc, file_name)
        else:
            print("Chunk conversion skipped")

        ## Creating Embeddings ##
            
        os.environ["OPENAI_API_KEY"]="Your OPENAI API key"

        EMBEDDING_MODEL='text-embedding-ada-002'
        embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL)
            
        if text_chunked != []:
            print("Chroma embeddings started")
            dbs_folder_path = os.path.join("./" + docFolder + "/" + userFolder + "/" + pdfFolder, "dbs")
            if not os.path.exists(dbs_folder_path):
                os.makedirs(dbs_folder_path)
            dbs_path=os.path.join(dbs_folder_path, "index")
            print(dbs_path, "dbs_path1")

            # Creating vector stores and storing in in that path using persist directory parameter
            dbs=Chroma.from_documents(text_chunked, embedding=embeddings,persist_directory=dbs_path) #basic operations 
            dbs.persist()
        else:
            print("Chroma embeddings skipped")

        if (os.path.exists(folder_path + "\\" + "dbs" + "\\" + "index") and text_chunked == []):
            print("Chroma db fetching started")
            dbs_path = "./" + docFolder + "/" + userFolder + "/" + pdfFolder + "/" + "dbs/index"
            print(dbs_path, "dbs_path2")

            dbs=Chroma(persist_directory=dbs_path, embedding_function=embeddings)
            dbs.persist()
        else:
            print("Chroma db fetching skipped")

        ##  Chat Model Defining ##
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0 ,max_tokens=100)

        ## Prompt Defining ##
        prompt_template=r"""Generate a descriptive answer from multiple references within the long text.
            If you are not able to generate the answer, just say not able to generate answer for the question from the given context.
            Consider the context that was given and do not think out of the context to generate the answer.
            {context}
            Question: {question}
            Answer:
            """ 
        # dbs.similarity_search_with_score(query="What is AI?")

        ret_db = dbs

        prompt=PromptTemplate(
            input_variables=['context','question'],
            template=prompt_template
        )    

        ## Generating Responses ##
        retriever=ret_db.as_retriever(search_type='similarity',search_kwargs={'k':3})
        memory1 = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,
                                                    combine_docs_chain_kwargs={"prompt": prompt},
                                                    #combine_docs_chain=doc_chain,
                                                        memory=memory1)  

        result=chain({"question":query}) 
        print(result['answer'])        

        
        
        response = json.dumps({
            "status" : "OKAY",
            "message" : result['answer']

        })
        
        return response

    except Exception as e:

        print("Error: ", e)
        response = json.dumps({
            "error": "{}".format(str(e))
        })
        return response


# Start with flask web app with debug as
# True only if this is the starting page
if(__name__ == "__main__"):
    app.run(debug=True, port=6000)