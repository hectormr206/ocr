# OCR Microservice

A high-performance FastAPI microservice for image preprocessing and OCR (Optical Character Recognition) with advanced image enhancement capabilities.

## 🚀 Features

- **Advanced Image Preprocessing**: Automatic deskew, perspective correction, and adaptive thresholding
- **OCR Integration**: Built-in Tesseract OCR with LSTM engine for accurate text extraction
- **Multi-language Support**: Configurable language support (default: Spanish)
- **RESTful API**: Clean endpoints for image processing and text extraction
- **Docker Ready**: Optimized containerization for easy deployment
- **Production Ready**: Health checks, error handling, and comprehensive logging

## 📋 Requirements

- Python 3.10+
- OpenCV
- Tesseract OCR
- FastAPI
- Docker (for containerized deployment)

## 🛠️ Installation

### Local Development

1. **Clone the repository:**

```bash
git clone https://github.com/hectormr206/ocr.git
cd ocr
```

2. **Install system dependencies:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa libgl1-mesa-glx

# macOS
brew install tesseract tesseract-lang
```

3. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the application:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **Build the image:**

```bash
docker build -t ocr-microservice .
```

2. **Run the container:**

```bash
docker run -p 8000:8000 ocr-microservice
```

## 🌐 API Endpoints

### Base URL

```
http://localhost:8000
```

### 1. Process Image

**POST** `/process-image`

Preprocesses an image for optimal OCR performance.

**Request:**

- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**

- Content-Type: `image/png`
- Body: Processed image binary

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/process-image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.jpg"
```

### 2. Extract Text

**POST** `/ocr-text`

Extracts text from an image using OCR.

**Parameters:**

- `file`: Image file (required)
- `lang`: Language code (optional, default: "spa")

**Response:**

```json
{
  "text": "Extracted text content...",
  "data": {
    "level": [1, 2, 3, ...],
    "page_num": [1, 1, 1, ...],
    "block_num": [0, 0, 0, ...],
    "par_num": [0, 0, 0, ...],
    "line_num": [0, 0, 0, ...],
    "word_num": [0, 1, 2, ...],
    "left": [10, 20, 30, ...],
    "top": [10, 10, 10, ...],
    "width": [100, 200, 300, ...],
    "height": [20, 20, 20, ...],
    "conf": [90, 85, 95, ...],
    "text": ["Invoice", "Total:", "$100.00"]
  }
}
```

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/ocr-text" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.jpg" \
  -F "lang=spa"
```

### 3. API Documentation

**GET** `/docs`

Interactive API documentation (Swagger UI)

**GET** `/redoc`

Alternative API documentation

## 🔧 Configuration

### Environment Variables

| Variable | Default   | Description      |
| -------- | --------- | ---------------- |
| `PORT`   | `8000`    | Application port |
| `HOST`   | `0.0.0.0` | Application host |

### Language Support

The service supports multiple languages through Tesseract. Common language codes:

- `spa` - Spanish (default)
- `eng` - English
- `fra` - French
- `deu` - German
- `por` - Portuguese

## 🚀 Deployment

### Coolify Deployment

1. **Push to GitHub:**

```bash
git add .
git commit -m "Initial OCR microservice"
git push origin main
```

2. **Configure in Coolify:**

   - Repository URL: `https://github.com/hectormr206/ocr`
   - Build Pack: `Dockerfile`
   - Port: `8000`
   - Environment Variables: `PORT=8001`

3. **Deploy:**
   - Coolify will automatically build and deploy your application
   - Access via: `https://yourdomain.com:PORT`

### Docker Compose

```yaml
version: "3.8"
services:
  ocr:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
    restart: unless-stopped
```

## 🔄 Integration Examples

### n8n Workflow Integration

1. **HTTP Request Node:**

   - Method: `POST`
   - URL: `https://yourdomain.com/process-image`
   - Response Format: `File`
   - Binary Property: `data` → File Parameter `file`

2. **Google Vision OCR Node:**
   - Use the processed image from the previous step
   - Connect to your OCR workflow

### Python Client Example

```python
import requests

def process_image(image_path, base_url="http://localhost:8000"):
    """Process an image through the OCR microservice"""

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{base_url}/process-image", files=files)

    if response.status_code == 200:
        with open('processed_image.png', 'wb') as f:
            f.write(response.content)
        return 'processed_image.png'
    else:
        raise Exception(f"Error: {response.status_code}")

def extract_text(image_path, lang="spa", base_url="http://localhost:8000"):
    """Extract text from an image"""

    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'lang': lang}
        response = requests.post(f"{base_url}/ocr-text", files=files, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}")

# Usage
processed_image = process_image("invoice.jpg")
text_data = extract_text(processed_image, lang="spa")
print(text_data['text'])
```

## 🏗️ Architecture

### Image Processing Pipeline

1. **Deskew**: Automatic rotation correction
2. **Perspective Warp**: Document alignment and correction
3. **Adaptive Thresholding**: Optimal binarization for OCR
4. **Text Extraction**: Tesseract LSTM engine

### Technology Stack

- **FastAPI**: High-performance web framework
- **OpenCV**: Computer vision and image processing
- **Tesseract**: OCR engine with LSTM
- **Pillow**: Image manipulation
- **Uvicorn**: ASGI server
- **Docker**: Containerization

## 📊 Performance

- **Processing Time**: ~2-5 seconds per image (depending on size)
- **Memory Usage**: ~200-500MB per container
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Image Formats**: JPEG, PNG, BMP, TIFF

## 🔍 Troubleshooting

### Common Issues

1. **Tesseract not found:**

   ```bash
   sudo apt-get install tesseract-ocr tesseract-ocr-spa
   ```

2. **OpenCV dependencies:**

   ```bash
   sudo apt-get install libgl1-mesa-glx
   ```

3. **Port already in use:**
   - Change `PORT` environment variable
   - Or kill existing process: `sudo lsof -ti:8000 | xargs kill -9`

### Health Check

```bash
curl http://localhost:8000/docs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [Tesseract](https://github.com/tesseract-ocr/tesseract) for OCR functionality
- [Coolify](https://coolify.io/) for easy deployment

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the troubleshooting section above

---

**Made with ❤️ for efficient OCR processing**
