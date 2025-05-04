import uvicorn
import os
from api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))
    uvicorn.run(app, host="0.0.0.0", port=port)
