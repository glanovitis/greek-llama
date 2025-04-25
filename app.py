import os
from dotenv import load_dotenv
import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
import time
import base64

load_dotenv()

st.set_page_config(page_title="Greek Classics Q&A", page_icon="🔍", layout="wide")


# Define colors - deep burgundy/terracotta
text_color = "#8B0000"  # Dark red/burgundy
header_color = "#654321"  # Darker brown for headers

# Add this at the beginning of your Streamlit app, right after importing streamlit
import streamlit as st

# Global style to change all text color
st.markdown("""
<style>
    /* Change all regular text */
    body, p, span, label, .stMarkdown, .stText {
        color: #000000 !important;
        font-family: 'Garamond', 'Georgia', serif !important;
        font-weight: bold !important;
    }

    /* Change all headers */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-family: 'Garamond', 'Georgia', serif!important;
    }

    /* Change sidebar text */
    .css-1d391kg, .css-hxt7ib {
        color: #000000 !important;
        font-family: 'Garamond', 'Georgia', serif!important;
    }

    /* Change button text */
    .stButton button {
        color: #000000 !important;
        font-family: 'Garamond', 'Georgia', serif!important;
    }

    /* Change input text */
    .stTextInput input, .stTextArea textarea {
        color: #000000 !important;
        font-family: 'Garamond', 'Georgia', serif!important;
    }
</style>
""", unsafe_allow_html=True)


# Background image function
def get_base64_of_bin_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(local_img_path):
    bin_str = get_base64_of_bin_file(local_img_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ⬅️ Place this BEFORE any Streamlit UI components
set_background("./mythic_background.jpg")


# --- 4. Big white heading ---
st.markdown("<h1 style='color: #ffffff!important; font-size: 60px;'> 🏛️ Oracle of the Epics 🏛️ </h1>", unsafe_allow_html=True)


# Page configuration

#st.title("**The Heroes' Quest** \n **Unravel the Epics of Iliad, Odyssey, and Aeneid**")

# Sidebar for configuration and information
with st.sidebar:
    st.header("About")
    st.write("This app allows you to ask questions about your documents using LlamaIndex and Groq.")

    st.header("Settings")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key")
        model_choice = st.selectbox(
            "Select Groq Model",
            ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
            index=0,
        )

    # Advanced settings expandable section
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=800)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=150)
        top_k = st.slider("Number of Retrieved Documents", min_value=1, max_value=10, value=2)


# Main function to load index and create query engine
@st.cache_resource
def initialize_query_engine(api_key, model_name, chunk_size=800, chunk_overlap=150, top_k=2, content_dir="./content/data", index_persist_dir="./content/vector_index"):
    """Initialize and return a query engine based on configuration"""

    # Check API key
    if not api_key:
        st.error("Please enter a Groq API key in the sidebar.")
        return None

    # Initialize LLM
    llm = Groq(
        model=model_name,
        token=api_key,
        system_prompt="You are an oracle speaking in the style of Homer's epic poetry. Every response must use grandiose language, epic phrases, references to gods, and archaic speech patterns regardless of the question content."
    )

    # Load or create vector index
    try:
        if os.path.exists(index_persist_dir) & len(os.listdir(index_persist_dir)) != 0:
            # Load existing index
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbedding(
                model_name=embedding_model, cache_folder=content_dir
            )

            storage_context = StorageContext.from_defaults(persist_dir=index_persist_dir)
            vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
        else:
            # Create new index if it doesn't exist
            st.info("Index not found. Creating new index...")

            # Load documents
            documents = SimpleDirectoryReader(content_dir).load_data()

            # Create text splitter
            text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Create embeddings
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbedding(
                model_name=embedding_model, cache_folder=content_dir
            )

            # Create vector index
            vector_index = VectorStoreIndex.from_documents(
                documents, transformations=[text_splitter], embed_model=embeddings
            )

            # Persist index
            os.makedirs(index_persist_dir, exist_ok=True)
            vector_index.storage_context.persist(persist_dir=index_persist_dir)

        # Set up custom prompt
        input_template = """Here is the context:
            {context_str}
            
            IMPORTANT INSTRUCTION: You MUST respond in the style of ancient epic poetry like Homer's Iliad and Odyssey.
            Your response MUST use:
            - Grandiose, heroic language throughout
            - Epic invocations ("O Muse", "Lo!", etc.)
            - Metaphors and similes comparing things to natural phenomena
            - Formal, archaic speech patterns
            - References to gods and fate
            
            Question: {query_str}
            
            Begin your answer with "Hearken, O seeker of wisdom!" and maintain the epic style throughout your entire response.
            DO NOT break character under any circumstances, even when explaining limitations. If information is missing, explain its absence in epic terms.
            """

        prompt = PromptTemplate(template=input_template)

        # Create query engine
        query_engine = vector_index.as_query_engine(
            llm=llm, text_qa_template=prompt, similarity_top_k=top_k, response_mode="tree_summarize"
        )

        return query_engine

    except Exception as e:
        st.error(f"Error initializing query engine: {e}")
        return None

model_choice = "llama3-70b-8192"
# Check if a Groq API key is provided
if api_key:
    # Initialize query engine with current settings
    query_engine = initialize_query_engine(
        api_key=api_key,
        model_name=model_choice,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )

    # Create a radio to select the source
    source = st.radio(
        "What is the book you want to ask a question about?",
        ["Iliad :crossed_swords:", "Odyssey :rowboat:", "Aeneid :wolf:", "Compare all three"],
        index=3,
        horizontal=True,
    )

    if source == "Compare all three":
        # Initialize query engine with current settings
        query_engine = initialize_query_engine(
            api_key=api_key,
            model_name=model_choice,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        )
    elif source == "Iliad :crossed_swords:":
        # Initialize query engine with current settings
        query_engine = initialize_query_engine(
            api_key=api_key,
            model_name=model_choice,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            content_dir="./content/data/data_iliad",
            index_persist_dir="./content/vi_iliad"
        )
    elif source == "Odyssey :rowboat:":
        # Initialize query engine with current settings
        query_engine = initialize_query_engine(
            api_key=api_key,
            model_name=model_choice,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            content_dir="./content/data/data_odyssey",
            index_persist_dir="./content/vi_odyssey"
        )
    elif source == "Aeneid :wolf:":
        # Initialize query engine with current settings
        query_engine = initialize_query_engine(
            api_key=api_key,
            model_name=model_choice,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            content_dir="./content/data/data_aeneid",
            index_persist_dir="./content/vi_aeneid"
        )

    # Main app logic
    if query_engine:
        # User input
        user_question = st.text_input("Explore the epic tales and ask your mythic question:", placeholder="e.g., Who is Penelope?")

        if user_question:
            with st.spinner("Consulting the oracles..."):
                try:
                    start_time = time.time()
                    answer = query_engine.query(user_question)
                    end_time = time.time()

                    # Display answer
                    st.header("Speak, O Muse, and grant thy oracle to the mortal's query")
                    st.caption(answer)

                    # Show source documents if available
                    if hasattr(answer, 'source_nodes') and answer.source_nodes:
                        with st.expander("Source Documents"):
                            for i, node in enumerate(answer.source_nodes):
                                st.markdown(f"**Source {i + 1}**")
                                st.text(node.node.text[:500] + "..." if len(node.node.text) > 500 else node.node.text)
                                st.divider()

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.info("Please try again or reformulate your question")
else:
    st.info("Please enter your Groq API key in the sidebar to get started.")

    # Show instructions if no API key is provided
    st.header("How to use this app")
    st.markdown("""
    1. Enter your Groq API key in the sidebar
    2. Adjust any advanced settings if needed
    3. Type your question in the text input
    4. Get answers based on your documents!
    """)

# Footer
st.divider()
st.caption("Built with LlamaIndex and Groq | Powered by Streamlit")