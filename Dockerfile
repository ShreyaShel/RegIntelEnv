# Use the official Hugging Face GPU-optimized base image (stable stack)
FROM huggingface/transformers-pytorch-gpu:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Frontend dependencies
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Back to project root
WORKDIR /app

# Expose ports (Backend: 7860, Frontend: 5173)
# Note: Hugging Face Spaces usually only exposes ONE port (7860).
# We will use a proxy or just serve the built frontend from FastAPI.
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]