import uvicorn
import os
import logging
from api import app

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Get port from environment variable
    port = int(os.environ.get("PORT", 8502))
    
    # Start the server with increased timeout
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=300,  # Increase keep-alive timeout
        log_level="info"
    )
