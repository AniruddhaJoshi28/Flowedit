import os
import sys
import argparse
import uvicorn

# Add the project root to the Python path so it can find the 'flowedit' module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flowedit.utils.env import load_flowedit_env
load_flowedit_env()

def main():
    parser = argparse.ArgumentParser(description="Start FlowEdit FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting FlowEdit API on http://{args.host}:{args.port}")
    print(f"Swagger UI will be available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run("flowedit.api.main:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
