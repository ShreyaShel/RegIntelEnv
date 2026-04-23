import subprocess
import threading
import os
import sys
import time
import platform

def run_backend():
    print("🚀 Starting FastAPI Backend on port 7860...")
    shell = platform.system() == "Windows"
    try:
        subprocess.run(["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"], shell=shell)
    except Exception as e:
        print(f"❌ Backend failed to start: {e}")

def run_frontend():
    print("🎨 Starting Vite Frontend on port 5173...")
    frontend_dir = os.path.join(os.getcwd(), "frontend")
    shell = platform.system() == "Windows"
    
    # Check if node_modules exists, if not run npm install
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("📦 node_modules not found, installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, shell=shell)
    
    try:
        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir, shell=shell)
    except Exception as e:
        print(f"❌ Frontend failed to start: {e}")

if __name__ == "__main__":
    try:
        # Create threads
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        frontend_thread = threading.Thread(target=run_frontend, daemon=True)

        # Start execution
        backend_thread.start()
        time.sleep(2) # Wait for backend port binding
        
        # Only start Vite dev server if not in a restricted environment like Hugging Face
        # Hugging Face Spaces typically define spaces-specific env vars
        is_huggingface = os.getenv("SPACE_ID") is not None
        if not is_huggingface:
            frontend_thread.start()
        else:
            print("📦 Production Environment Detected: Serving built frontend from Backend...")

        # Check for AI Intelligence Fuel
        has_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
        
        print("\n" + "═"*50)
        print("🌐 REG-INTEL-ENV SYSTEM LAUNCHED")
        print(f"🖥️  Neural Dashboard: http://localhost:5173")
        print(f"⚙️  Backend Docs:    http://localhost:7860/docs")
        if has_key:
            print(f"🧠 AI GRADER:        ENABLED (Using HF Inference Endpoint)")
        else:
            print(f"🧠 AI GRADER:        FALLBACK (Simulation Mode - HF_TOKEN not found)")
        print("═"*50 + "\n")

        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down environment...")
        sys.exit(0)
