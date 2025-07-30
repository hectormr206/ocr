FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de la aplicación
WORKDIR /app
COPY main.py ./
COPY requirements.txt ./

# Instalar paquetes Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto y ejecutar
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 