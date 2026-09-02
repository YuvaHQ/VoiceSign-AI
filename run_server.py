"""
run_server.py
-------------
Launch the SignBridge FastAPI web application and WebSocket server.
Usage:
    python run_server.py [--host 127.0.0.1] [--port 8000] [--reload]
"""

import argparse
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the SignBridge Sign Language Application Server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" 🤟 Starting SignBridge — Multilingual & Personalized Sign Language System")
    print(f" 🌐 Web UI & API available at: http://{args.host}:{args.port}")
    print(f" 📖 Interactive API Docs at:  http://{args.host}:{args.port}/docs")
    print("=" * 70 + "\n")

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()