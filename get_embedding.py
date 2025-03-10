from typing import Optional
from langchain_ollama import OllamaEmbeddings
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "mxbai-embed-large"  # Better for general text
ALTERNATIVE_MODEL = "nomic-embed-text"  # Better for technical content
MODEL_CACHE = {}


def get_embedding(model_name: Optional[str] = None) -> OllamaEmbeddings:
    """
    Get an embedding function using Ollama with caching for better performance.

    Args:
        model_name (Optional[str]): The name of the model to use. 
                                  If None, uses the default model.

    Returns:
        OllamaEmbeddings: The embedding function instance

    Note:
        Available models:
        - mxbai-embed-large (default): Better for general text understanding
        - nomic-embed-text (alternative): Better for technical content

    Raises:
        Exception: If there's an error initializing the embedding function
    """
    try:
        model = model_name or DEFAULT_MODEL

        # Use cached model if available
        if model in MODEL_CACHE:
            logger.debug(f"Using cached embedding model: {model}")
            return MODEL_CACHE[model]

        # Initialize new model
        logger.info(f"Initializing new embedding model: {model}")
        embeddings = OllamaEmbeddings(
            model=model
        )

        # Cache the model
        MODEL_CACHE[model] = embeddings
        return embeddings

    except Exception as e:
        logger.error(f"Error initializing embedding function: {str(e)}")
        raise
