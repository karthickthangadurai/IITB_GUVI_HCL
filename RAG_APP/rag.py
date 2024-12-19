# from langchain_core.prompts import ChatPromptTemplate
# # from langchain_ollama import OllamaLLM
# from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os

# class SimpleRAG:
#     def __init__(self, ollama_model="llama2"):
#         """Initialize the RAG system with an Ollama model."""
#         self.llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key="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")
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'}
#         )
        
#     def load_pdf(self, pdf_path):
#         """Load and process a PDF document."""
#         # Load and split the PDF
#         docs = PyPDFLoader(pdf_path).load()
#         texts = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         ).split_documents(docs)
        
#         # Create or load vector store
#         index_path = f"faiss_index/{os.path.basename(pdf_path)}_index"
#         if os.path.exists(index_path):
#             self.vector_store = FAISS.load_local(index_path, self.embeddings)
#         else:
#             self.vector_store = FAISS.from_documents(texts, self.embeddings)
#             os.makedirs("faiss_index", exist_ok=True)
#             self.vector_store.save_local(index_path)
    
#     def get_answer(self, question, use_context=True):
#         """Get answer from the system."""
#         # Create the appropriate prompt template
#         if use_context and hasattr(self, 'vector_store'):
#             # Get relevant documents
#             docs = self.vector_store.similarity_search(question, k=4)
#             context = "\n\n".join(doc.page_content for doc in docs)
            
#             prompt = ChatPromptTemplate.from_template("""
#             Answer based on the following context. If the context isn't relevant, use your knowledge.
            
#             Context: {context}
#             Question: {question}
            
#             Let's think about this step by step:
#             """)
            
#             # Get answer with context
#             chain_input = {"context": context, "question": question}
#         else:
#             # Use simple prompt without context
#             prompt = ChatPromptTemplate.from_template("""
#             Answer the following question:
#             Question: {question}
            
#             Let's think about this step by step:
#             """)
            
#             chain_input = {"question": question}
        
#         # Create and run the chain
#         chain = prompt | self.llm | StrOutputParser()
#         return chain.invoke(chain_input)

# # Example usage
# if __name__ == "__main__":
#     # Initialize the system
#     rag = SimpleRAG()
    
#     # Example with PDF context
#     # rag.load_pdf("path_to_your.pdf")  # Optional
#     # answer = rag.get_answer("What is discussed in the document?")
#     # print("Answer:", answer)
    
#     # Example without PDF context
#     answer = rag.get_answer("What is LangChain?", use_context=False)
#     print("Answer:", answer) 



from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def initialize_models():
    """Initialize and return LLM and embeddings models."""
    llm = ChatGroq(
        model_name="mixtral-8x7b-32768", 
        groq_api_key="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"
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return llm, embeddings

def process_pdf(pdf_path, embeddings):
    """Load and process a PDF document, return vector store."""
    # Load and split the PDF
    docs = PyPDFLoader(pdf_path).load()
    texts = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(docs)
    
    # Create or load vector store
    index_path = f"faiss_index/{os.path.basename(pdf_path)}_index"
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization = True)
    else:
        vector_store = FAISS.from_documents(texts, embeddings)
        os.makedirs("faiss_index", exist_ok=True)
        vector_store.save_local(index_path)
    
    return vector_store

def get_answer(question, llm, vector_store=None):
    """Get answer from the system with optional context."""
    if vector_store:
        # Get relevant documents
        docs = vector_store.similarity_search(question, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        prompt = ChatPromptTemplate.from_template("""
        Answer based on the following context. If the context isn't relevant, use your knowledge.
        
        Context: {context}
        Question: {question}
        
        Let's think about this step by step:
        """)
        
        chain_input = {"context": context, "question": question}
    else:
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question:
        Question: {question}
        
        Let's think about this step by step:
        """)
        
        chain_input = {"question": question}
    
    # Create and run the chain
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(chain_input)

def main():
    # Initialize models
    llm, embeddings = initialize_models()
    
    # Example without PDF context
    answer = get_answer("What is LangChain?", llm)
    print("Answer without context:", answer)
    
    # Example with PDF context
    # pdf_path = "path_to_your.pdf"
    # vector_store = process_pdf(pdf_path, embeddings)
    # answer = get_answer("What is discussed in the document?", llm, vector_store)
    # print("Answer with context:", answer)

if __name__ == "__main__":
    main()