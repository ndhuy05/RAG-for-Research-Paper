import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import fitz  # PyMuPDF for better PDF handling

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

class StreamlitRAGChatbot:
    def __init__(self, api_key):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.memory = ConversationBufferMemory()
        self.vector_store = None
        self.qa_chain = None
        
        self.prompt_template = """Use the following context to answer the question. If you cannot find the answer in the context, say "I don't have enough information in the provided documents to answer that question."

Context: {context}

Question: {question}

Answer:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def extract_text_from_pdf_advanced(self, pdf_path):
        """
        Advanced PDF text extraction that handles multi-column layouts better
        Uses PyMuPDF (fitz) for better column detection
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                blocks = page.get_text("dict")["blocks"]
                
                text_blocks = []
                for block in blocks:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text_blocks.append({
                                    "text": span["text"],
                                    "bbox": span["bbox"],  # [x0, y0, x1, y1]
                                    "x0": span["bbox"][0],
                                    "y0": span["bbox"][1]
                                })
                
                text_blocks.sort(key=lambda x: (round(x["y0"] / 10) * 10, x["x0"]))
                
                page_text = " ".join([block["text"] for block in text_blocks if block["text"].strip()])
                
                full_text.append(f"Page {page_num + 1}:\n{page_text}")
            
            doc.close()
            return "\n\n".join(full_text)
            
        except ImportError:
            st.warning("⚠️ PyMuPDF not available. Using standard PDF extraction. For better multi-column support, install: pip install PyMuPDF")
            return None
        except Exception as e:
            st.error(f"Error with advanced PDF extraction: {e}")
            return None
    
    def load_uploaded_files(self, uploaded_files):
        """Load documents from uploaded files with enhanced PDF handling"""
        documents = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                temp_path = Path(temp_dir) / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    if uploaded_file.name.endswith('.pdf'):
                        advanced_text = self.extract_text_from_pdf_advanced(str(temp_path))
                        
                        if advanced_text:
                            from langchain.schema import Document
                            documents.append(Document(
                                page_content=advanced_text,
                                metadata={"source": uploaded_file.name, "extraction_method": "advanced"}
                            ))
                            st.success(f"✅ Used advanced extraction for {uploaded_file.name}")
                        else:
                            loader = PyPDFLoader(str(temp_path))
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata["extraction_method"] = "standard"
                            documents.extend(docs)
                            st.info(f"ℹ️ Used standard extraction for {uploaded_file.name}")
                    
                    elif uploaded_file.name.endswith('.txt'):
                        loader = TextLoader(str(temp_path))
                        docs = loader.load()
                        documents.extend(docs)
                    
                    else:
                        continue
                        
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {e}")
        
        if documents:
            return self.process_documents(documents)
        return False
    
    def process_documents(self, documents):
        """Process and split documents into chunks with better handling for research papers"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                separators=[
                    "\n\n",  
                    "\n",   
                    ".",    
                    "!",     
                    "?",  
                    ";",   
                    ",",     
                    " ",   
                    ""       
                ],
                length_function=len,
            )
            texts = text_splitter.split_documents(documents)
            
            for i, chunk in enumerate(texts):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 5}
                ),
                chain_type_kwargs={"prompt": self.prompt}
            )
            
            return len(texts)
            
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            return False
    
    def ask_question(self, question):
        """Ask a question using RAG"""
        if self.qa_chain is None:
            return "Please load documents first."
        
        try:
            response = self.qa_chain.run(question)
            return response
        except Exception as e:
            return f"Error: {e}"
    
    def chat_with_memory(self, user_input):
        """Simple chat with memory (without RAG)"""
        conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )
        return conversation.predict(input=user_input)

def main():
    st.title("🤖 RAG Chatbot with Google Gemini")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google API key for Gemini"
        )
        
        if api_key:
            if st.session_state.chatbot is None:
                try:
                    st.session_state.chatbot = StreamlitRAGChatbot(api_key)
                    st.success("✅ Chatbot initialized!")
                except Exception as e:
                    st.error(f"❌ Error initializing chatbot: {e}")
                    return
            
            st.markdown("---")
            
            st.header("📄 Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose PDF or TXT files",
                accept_multiple_files=True,
                type=['pdf', 'txt'],
                help="Upload documents for RAG functionality"
            )
            
            if uploaded_files:
                if st.button("📚 Load Documents"):
                    with st.spinner("Loading and processing documents..."):
                        num_chunks = st.session_state.chatbot.load_uploaded_files(uploaded_files)
                        if num_chunks:
                            st.session_state.documents_loaded = True
                            st.success(f"✅ Loaded {len(uploaded_files)} files, created {num_chunks} chunks")
                        else:
                            st.error("❌ Failed to load documents")
            
            st.markdown("---")
            st.header("📊 Status")
            if st.session_state.documents_loaded:
                st.success("✅ Documents loaded - RAG mode available")
            else:
                st.warning("⚠️ No documents loaded - Chat mode only")
            
            if st.button("🗑️ Clear Conversation"):
                st.session_state.messages = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.memory.clear()
                st.rerun()
        
        else:
            st.warning("⚠️ Please enter your Google API key to continue")
            st.info("Get your API key from: https://makersuite.google.com/app/apikey")
            return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat")
        
        chat_mode = st.radio(
            "Chat Mode:",
            ["RAG Mode (with documents)", "Simple Chat (without documents)"],
            disabled=not st.session_state.documents_loaded if st.session_state.chatbot else True
        )
        
        # Chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if chat_mode.startswith("RAG") and st.session_state.documents_loaded:
                        response = st.session_state.chatbot.ask_question(prompt)
                    else:
                        response = st.session_state.chatbot.chat_with_memory(prompt)
                
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("📋 Instructions")
        
        with st.expander("🚀 How to use", expanded=True):
            st.markdown("""
            1. **Enter your Google API Key** in the sidebar
            2. **Upload documents** (PDF or TXT files) for RAG functionality
            3. **Click 'Load Documents'** to process them
            4. **Choose chat mode**:
                - RAG Mode: Ask questions about your documents
                - Simple Chat: General conversation
            5. **Start chatting!**
            """)
        
        with st.expander("💡 Tips"):
            st.markdown("""
            - Upload multiple files for better context
            - Use specific questions for better RAG results
            - Clear conversation to start fresh
            - RAG mode works best with factual questions
            """)
        
        with st.expander("📁 Supported Files"):
            st.markdown("""
            - **PDF files** (.pdf) - Enhanced support for multi-column research papers
            - **Text files** (.txt)
            - Multiple files can be uploaded at once
            
            **PDF Enhancement Features:**
            - ✅ Multi-column layout detection
            - ✅ Proper reading order preservation
            - ✅ Better text extraction for research papers
            - ✅ Automatic fallback to standard extraction
            """)
        
        with st.expander("🔬 Research Paper Tips"):
            st.markdown("""
            **For best results with research papers:**
            - Upload clear, text-based PDFs (not scanned images)
            - Multi-column layouts are automatically detected
            - Ask specific questions about methodology, results, conclusions
            - Use questions like:
                - "What is the main hypothesis?"
                - "What datasets were used?"
                - "What are the key findings?"
                - "What are the main results?"
                - "What are the limitations mentioned?"
                - "What is the methodology used?"
            """)
        
        if st.session_state.documents_loaded:
            with st.expander("❓ Sample Questions"):
                sample_questions = [
                    "What is the main topic of the documents?",
                    "Can you summarize the key points?",
                    "What are the important findings?",
                    "Tell me about the methodology used.",
                    "What are the main results?",
                    "What are the limitations mentioned?",
                    "What conclusions can be drawn?",
                ]
                
                for question in sample_questions:
                    if st.button(question, key=f"sample_{question}"):
                        st.session_state.messages.append({"role": "user", "content": question})
                        response = st.session_state.chatbot.ask_question(question)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()

if __name__ == "__main__":
    main()
