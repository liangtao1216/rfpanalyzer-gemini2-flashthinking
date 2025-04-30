import os
import streamlit as st
import pandas as pd
from docx import Document
import io
import re
from typing import Tuple, List, Optional
import logging
from pathlib import Path
from llama_parse import LlamaParse
from dotenv import load_dotenv
import tempfile
import requests
import json
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API keys from environment variables
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not LLAMA_CLOUD_API_KEY:
    raise ValueError("Missing required API keys in environment variables")

# Initialize LlamaParse
parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown",
    verbose=True
)

# Configure local model settings
MODEL_NAME = "qwen2.5:latest"

# Initialize OpenAI client for Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"  # Ollama doesn't need an API key
)

def check_model_availability():
    """Check if the model is available and running"""
    try:
        logger.info("Checking model availability...")
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models = response.json().get('models', [])
        available_models = [model['name'] for model in models]
        
        if MODEL_NAME not in available_models:
            logger.error(f"Model {MODEL_NAME} is not available. Available models: {available_models}")
            return False
            
        logger.info(f"Model {MODEL_NAME} is available")
        return True
    except Exception as e:
        logger.error(f"Failed to check model availability: {e}")
        return False

# Check model availability at startup
if not check_model_availability():
    raise RuntimeError(f"Model {MODEL_NAME} is not available. Please make sure Ollama is running and the model is pulled.")

def generate_response(prompt: str) -> str:
    """
    Generate response using local Qwen model through Ollama with OpenAI client
    """
    try:
        logger.info(f"Sending prompt to local Qwen model")
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.95,
            max_tokens=1000
        )
        
        logger.info("Received response from local Qwen model")
        
        if not response or not response.choices:
            logger.warning("Empty response received from local Qwen model")
            return None
            
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return None

def analyze_section(text, section_type="requirements"):
    """Analyze specific sections of the RFP"""
    logger.info(f"Starting {section_type} section analysis")
    
    prompts = {
        "requirements": """
            Analyze the technical requirements in this section:
            1. List all mandatory requirements
            2. Identify optional requirements
            3. Flag any unclear specifications
            4. Note any technical constraints
            """,
        "timeline": """
            Analyze the timeline aspects:
            1. List all key dates and deadlines
            2. Identify major milestones
            3. Flag any tight or unrealistic timelines
            4. Note dependencies between phases
            """,
        "compliance": """
            Analyze compliance requirements:
            1. List all regulatory requirements
            2. Identify certification needs
            3. Note security requirements
            4. Flag critical compliance issues
            """,
        "positions": """
            Analyze the staffing requirements in this section and provide a detailed markdown-formatted response covering:
            1. Extract all listed job positions or roles
            2. For each position, extract its full job description (JD), including:
                a. Responsibilities
                b. Required qualifications (e.g., certifications, education, experience, skills)
                c. Preferred or optional qualifications (e.g., certifications, languages)
                d. Keywords for resume matching
            3. Flag any ambiguous or unclear position descriptions
            """
    }
    
    try:
        prompt = f"{prompts.get(section_type, prompts['requirements'])}\n\nContent:\n{text}"
        response = generate_response(prompt)
        if not response:
            logger.warning(f"Empty response received for {section_type} analysis")
            return None
        return response
    except Exception as e:
        logger.error(f"Error in {section_type} analysis: {e}", exc_info=True)
        return None

def analyze_rfp_content(text: str) -> str:
    """
    Analyze the full content of the RFP document.
    """
    logger.info("Starting RFP content analysis")
    prompt = """
    Analyze this RFP document and provide a detailed markdown-formatted response covering:

    # Executive Summary
    - Brief overview of the RFP
    - Key objectives and goals

    # Requirements Analysis
    - ## Technical Requirements
      - Core technical specifications
      - Optional features
      - Integration requirements
    - ## Business Requirements
      - Mandatory business needs
      - Optional enhancements
    
    # Timeline and Milestones
    - Key dates and deadlines
    - Project phases
    - Dependencies

    # Risk Assessment
    - Technical risks
    - Business risks
    - Compliance concerns

    # Recommendations
    - Strategic approach
    - Key focus areas
    - Potential challenges

    Please analyze the following content:
    {text}
    """
    try:
        response = generate_response(prompt.format(text=text))
        if not response:
            logger.warning("Empty response received for RFP content analysis")
            return None
        return response
    except Exception as e:
        logger.error(f"Error analyzing RFP content: {e}", exc_info=True)
        return None

def extract_text_from_pdf(file) -> Tuple[str, List[pd.DataFrame]]:
    """
    Extract text from PDF file using LlamaParse
    
    Args:
        file: File-like object containing PDF
        
    Returns:
        Tuple[str, List[pd.DataFrame]]: Extracted text and tables
    """
    try:
        logger.info("Starting PDF extraction")
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
            logger.info(f"Created temporary file: {tmp_path}")

        # Parse the PDF using LlamaParse
        logger.info("Parsing PDF with LlamaParse")
        result = parser.load_data(tmp_path)
        logger.info("PDF parsing completed")
        
        # Combine all pages into one text
        logger.info("Extracting text from pages")
        text = "\n\n---\n\n".join([page.text for page in result])
        
        # Extract tables if available
        tables = []
        logger.info("Starting table extraction")
        for page in result:
            if hasattr(page, 'tables') and page.tables:
                for table in page.tables:
                    try:
                        df = pd.DataFrame(table)
                        if not df.empty:
                            tables.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to convert table to DataFrame: {e}")

        logger.info(f"Extracted {len(tables)} tables from PDF")

        # Clean up temporary file
        os.unlink(tmp_path)
        logger.info("Temporary file cleaned up")
            
        return text, tables
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to process PDF: {str(e)}")

def extract_text_from_docx(file):
    """Extract text and tables from DOCX file"""
    logger.info("Starting DOCX extraction")
    try:
        doc = Document(file)
        text = ""
        tables = []
        
        # Extract text
        logger.info("Extracting text from paragraphs")
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Extract tables
        logger.info("Starting table extraction from DOCX")
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                table_data.append(row_data)
            tables.append(pd.DataFrame(table_data[1:], columns=table_data[0]))
        
        logger.info(f"Extracted {len(tables)} tables from DOCX")
        return text, tables
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}", exc_info=True)
        raise

def extract_text_from_txt(file):
    """Extract text from TXT file"""
    logger.info("Starting TXT file extraction")
    try:
        text = file.getvalue().decode()
        logger.info("TXT file extraction completed")
        return text, []
    except Exception as e:
        logger.error(f"TXT extraction failed: {e}", exc_info=True)
        raise

def clean_text(text):
    """Clean extracted text"""
    logger.info("Starting text cleaning")
    try:
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        cleaned = text.strip()
        logger.info("Text cleaning completed")
        return cleaned
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}", exc_info=True)
        raise

def analyze_tables(tables):
    """Analyze tables found in the document"""
    analysis = []
    for i, table in enumerate(tables):
        if not table.empty:
            analysis.append(f"\nTable {i+1} Analysis:")
            analysis.append(f"- Columns: {', '.join(table.columns.tolist())}")
            analysis.append(f"- Rows: {len(table)}")
            analysis.append("- Content Summary:")
            for col in table.columns:
                unique_values = table[col].nunique()
                analysis.append(f"  * {col}: {unique_values} unique values")
    
    return "\n".join(analysis)

# Update the main function to include caching
@st.cache_data
def process_document(file_content, file_type: str) -> Tuple[str, List[pd.DataFrame]]:
    """Cache the document processing results"""
    logger.info(f"Starting document processing for file type: {file_type}")
    try:
        if file_type == 'pdf':
            logger.info("Processing PDF document")
            result = extract_text_from_pdf(file_content)
        elif file_type == 'docx':
            logger.info("Processing DOCX document")
            result = extract_text_from_docx(file_content)
        else:  # txt
            logger.info("Processing TXT document")
            result = extract_text_from_txt(file_content)
        logger.info("Document processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in process_document: {str(e)}")
        raise

def main():
    logger.info("Starting RFP Document Analyzer application")
    st.set_page_config(page_title="RFP Document Analyzer", layout="wide")
    
    # Check for API keys
    if not LLAMA_CLOUD_API_KEY:
        logger.error("Missing required API keys in environment variables!")
        st.error("Missing required API keys in environment variables!")
        st.stop()
    
    logger.info("API keys validated successfully")
    
    # Move controls to sidebar
    with st.sidebar:
        st.title("📄 RFP Document Analyzer")
        st.markdown("Upload your RFP document for analysis.")
        
        # File uploader with size limit
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        uploaded_file = st.file_uploader(
            "Choose an RFP document",
            type=['pdf', 'docx', 'txt'],
            help="Upload a PDF, DOCX, or TXT file (max 50MB)"
        )
        
        if uploaded_file:
            logger.info(f"File uploaded: {uploaded_file.name} (Size: {uploaded_file.size/1024:.2f} KB)")
            st.write(f"File: {uploaded_file.name}")
            st.write(f"Size: {uploaded_file.size/1024:.2f} KB")
            
            # Analysis control buttons
            analyze_full = st.button("🔍 Analyze Full Content", use_container_width=True)
            analyze_req = st.button("📋 Analyze Requirements", use_container_width=True)
            analyze_timeline = st.button("⏱️ Analyze Timeline", use_container_width=True)
            analyze_compliance = st.button("✓ Analyze Compliance", use_container_width=True)
            analyze_positions = st.button("👤 Analyze Job Descriptions", use_container_width=True)

    # Main content area
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE:
            logger.warning(f"File size ({uploaded_file.size/1024/1024:.2f}MB) exceeds limit of 50MB")
            st.error("File size exceeds 50MB limit. Please upload a smaller file.")
            return
            
        try:
            logger.info("Starting document processing")
            # Process document with caching
            text, tables = process_document(
                uploaded_file,
                uploaded_file.name.split('.')[-1].lower()
            )
            logger.info(f"Document processed successfully. Found {len(tables)} tables")
            
            # Clean extracted text
            logger.info("Cleaning extracted text")
            cleaned_text = clean_text(text)
            
            # Create two columns for better layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📑 Analysis Results")
                
                # Show analysis based on button clicks
                if analyze_full:
                    logger.info("Starting full content analysis")
                    with st.spinner("Analyzing full content..."):
                        analysis = analyze_rfp_content(cleaned_text)
                        logger.info("Full content analysis completed")
                        if analysis:
                            logger.info("Displaying analysis results")
                            st.markdown(analysis)
                        else:
                            logger.warning("No analysis results to display")
                            st.warning("No analysis results available. Please try again.")
                
                if analyze_req:
                    logger.info("Starting requirements analysis")
                    with st.spinner("Analyzing requirements..."):
                        req_analysis = analyze_section(cleaned_text, "requirements")
                        logger.info("Requirements analysis completed")
                        if req_analysis:
                            st.markdown(req_analysis)
                        else:
                            st.warning("No requirements analysis available. Please try again.")
                
                if analyze_timeline:
                    logger.info("Starting timeline analysis")
                    with st.spinner("Analyzing timeline..."):
                        timeline_analysis = analyze_section(cleaned_text, "timeline")
                        logger.info("Timeline analysis completed")
                        if timeline_analysis:
                            st.markdown(timeline_analysis)
                        else:
                            st.warning("No timeline analysis available. Please try again.")
                
                if analyze_compliance:
                    logger.info("Starting compliance analysis")
                    with st.spinner("Analyzing compliance..."):
                        compliance_analysis = analyze_section(cleaned_text, "compliance")
                        logger.info("Compliance analysis completed")
                        if compliance_analysis:
                            st.markdown(compliance_analysis)
                        else:
                            st.warning("No compliance analysis available. Please try again.")

                if analyze_positions:
                    logger.info("Starting positions analysis")
                    with st.spinner("Analyzing positions..."):
                        positions_analysis = analyze_section(cleaned_text, "positions")
                        logger.info("Positions analysis completed")
                        if positions_analysis:
                            st.markdown(positions_analysis)
                        else:
                            st.warning("No positions analysis available. Please try again.")
            
            with col2:
                # Show extracted text and tables
                with st.expander("📄 Extracted Text", expanded=False):
                    st.text_area("Content", cleaned_text, height=300)
                
                if tables:
                    logger.info(f"Displaying {len(tables)} extracted tables")
                    with st.expander("📊 Extracted Tables", expanded=False):
                        table_analysis = analyze_tables(tables)
                        st.markdown(table_analysis)
                        for i, table in enumerate(tables):
                            st.markdown(f"**Table {i+1}**")
                            st.dataframe(table, use_container_width=True)
                
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            st.error(f"Error processing document: {str(e)}")
            st.error("Please make sure the document is not corrupted and try again.")
    else:
        logger.info("No file uploaded - displaying welcome message")
        # Show welcome message when no file is uploaded
        st.markdown("""
        # Welcome to RFP Document Analyzer! 👋
        
        This tool helps you analyze Request for Proposal (RFP) documents by:
        
        - Extracting and analyzing content
        - Identifying key requirements
        - Analyzing timelines and milestones
        - Checking compliance requirements
        - Extracting job descriptions
        
        To get started, upload your RFP document using the sidebar.
        """)

if __name__ == "__main__":
    try:
        logger.info("Starting RFP Analyzer application")
        main()
        logger.info("Application completed successfully")
    except Exception as e:
        logger.error("Application crashed", exc_info=True)
        st.error("🚨 App crashed due to an internal error.")
        st.exception(e)  # This shows full traceback in Streamlit app
