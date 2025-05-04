import uvicorn
import os
import logging
from api import app, get_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Preload model before starting server
    logger.info("Preloading model before starting server...")
    try:
        get_model()
        logger.info("Model preloaded successfully!")
    except Exception as e:
        logger.error(f"Failed to preload model: {e}")
        logger.error("Continuing anyway...")
    
    # Start server
    port = int(os.environ.get("PORT", 8502))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
