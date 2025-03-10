import argparse
import os
import shutil
import re
import logging
from typing import List, Dict
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding import get_embedding
from langchain_chroma import Chroma
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data"
CHUNK_SIZE = 1000  # Increased chunk size for better context
CHUNK_OVERLAP = 200  # Increased overlap for better context continuity
MIN_CHUNK_SIZE = 100  # Minimum chunk size to avoid too small chunks


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing unnecessary newlines and extra spaces while preserving paragraphs.
    Also removes common noise patterns and normalizes text.

    Args:
        text (str): The input text to clean

    Returns:
        str: The cleaned text with preserved paragraph structure
    """
    # Remove common noise patterns
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'©.*?\d{4}', '', text)
    text = re.sub(r'www\..*?\.com', '', text)

    # Normalize whitespace
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []

    for para in paragraphs:
        # Replace single newlines with spaces within paragraphs
        cleaned_para = re.sub(r"(?<!\n)\n(?!\n)", " ", para)
        # Remove extra spaces
        cleaned_para = re.sub(r"\s{2,}", " ", cleaned_para)
        # Remove leading/trailing whitespace
        cleaned_para = cleaned_para.strip()
        if cleaned_para:  # Only keep non-empty paragraphs
            cleaned_paragraphs.append(cleaned_para)

    return "\n\n".join(cleaned_paragraphs)


def load_documents() -> List[Document]:
    """
    Load and clean documents from the data directory.
    Also extracts and adds metadata for better context.

    Returns:
        List[Document]: List of cleaned Document objects with enhanced metadata

    Raises:
        Exception: If there's an error loading documents
    """
    try:
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        raw_documents = document_loader.load()

        processed_documents = []
        for doc in raw_documents:
            # Clean the content
            doc.page_content = clean_text(doc.page_content)

            # Extract and enhance metadata
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0)

            # Add additional metadata
            doc.metadata.update({
                "source_type": os.path.splitext(source)[1][1:].upper(),
                "total_pages": len(raw_documents),
                "page_number": page,
                "processed_date": str(datetime.now()),
                "content_length": len(doc.page_content)
            })

            processed_documents.append(doc)

        logger.info(
            f"Successfully loaded and processed {len(processed_documents)} documents")
        return processed_documents

    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    Uses a more sophisticated splitting strategy.

    Args:
        documents (List[Document]): List of documents to split

    Returns:
        List[Document]: List of split document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ".", "!", "?", ",",
                    " ", ""],  # More granular separators
        keep_separator=True  # Keep separators for better context
    )

    chunks = text_splitter.split_documents(documents)

    # Filter out chunks that are too small
    chunks = [chunk for chunk in chunks if len(
        chunk.page_content) >= MIN_CHUNK_SIZE]

    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Calculate unique IDs for document chunks with enhanced metadata.

    Args:
        chunks (List[Document]): List of document chunks

    Returns:
        List[Document]: List of chunks with added IDs and enhanced metadata
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add enhanced metadata
        chunk.metadata.update({
            "id": chunk_id,
            "chunk_index": current_chunk_index,
            "chunk_size": len(chunk.page_content),
            "is_first_chunk": current_chunk_index == 0,
            "is_last_chunk": False  # Will be updated in next pass
        })

    # Update is_last_chunk flag
    for i, chunk in enumerate(chunks):
        if i == len(chunks) - 1 or chunk.metadata["source"] != chunks[i + 1].metadata["source"]:
            chunk.metadata["is_last_chunk"] = True

    return chunks


def add_to_chroma(chunks: List[Document]) -> None:
    """
    Add document chunks to the Chroma database.

    Args:
        chunks (List[Document]): List of document chunks to add

    Raises:
        Exception: If there's an error adding documents to the database
    """
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding()
        )

        chunks_with_ids = calculate_chunk_ids(chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = [
            chunk for chunk in chunks_with_ids
            if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            logger.info(
                f"Added {len(new_chunks)} new documents to the database")
        else:
            logger.info("No new documents to add")

    except Exception as e:
        logger.error(f"Error adding documents to Chroma: {str(e)}")
        raise


def clear_database() -> None:
    """Clear the Chroma database directory."""
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            logger.info("Successfully cleared the database")
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        raise


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Populate the RAG database with documents"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database before adding new documents"
    )
    args = parser.parse_args()

    try:
        if args.reset:
            clear_database()

        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)

    except Exception as e:
        logger.error(f"Failed to populate database: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
