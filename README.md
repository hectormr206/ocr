# OCR Microservice

A high-performance FastAPI microservice for image preprocessing and OCR (Optical Character Recognition) with advanced image enhancement capabilities. Supports both **Tesseract** and **PaddleOCR** engines for comparison and optimal results.

## 🚀 Features

- **Advanced Image Preprocessing**: Automatic deskew, perspective correction, and adaptive thresholding
- **Dual OCR Engines**: Tesseract and PaddleOCR for comparison and optimal results
- **OCR Integration**: Built-in Tesseract OCR with LSTM engine and PaddleOCR for accurate text extraction
- **Multi-language Support**: Configurable language support (default: Spanish)
- **RESTful API**: Clean endpoints for image processing and text extraction
- **Docker Ready**: Optimized containerization for easy deployment
- **Production Ready**: Health checks, error handling, and comprehensive logging
- **Performance Comparison**: Side-by-side comparison of OCR engines

## 📋 Requirements

- Python 3.10+
- OpenCV
- Tesseract OCR
- PaddleOCR
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

### 2. Extract Text (Tesseract)

**POST** `/ocr-text`

Extracts text from an image using Tesseract OCR.

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
  },
  "confidence": 85.5,
  "words_count": 45,
  "language": "spa"
}
```

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/ocr-text" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.jpg" \
  -F "lang=spa"
```

### 3. Complete OCR (Tesseract)

**POST** `/ocr-complete`

Processes image and extracts text in one step (similar to OCR.space).

**Parameters:**

- `file`: Image file (required)
- `lang`: Language code (optional, default: "spa")

**Response:**

```json
{
  "success": true,
  "text": "Extracted text content...",
  "confidence": 85.5,
  "words_count": 45,
  "language": "spa",
  "processing_time": "~2-5 seconds"
}
```

### 4. PaddleOCR

**POST** `/ocr-paddle`

Extracts text using PaddleOCR (no preprocessing required).

**Parameters:**

- `file`: Image file (required)

**Response:**

```json
{
  "success": true,
  "text": "Extracted text content...",
  "confidence": 92.1,
  "words_count": 47,
  "processing_time": "1.87 seconds",
  "engine": "PaddleOCR"
}
```

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/ocr-paddle" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.jpg"
```

### 5. OCR Comparison

**POST** `/ocr-comparison`

Compares Tesseract vs PaddleOCR results side-by-side.

**Parameters:**

- `file`: Image file (required)
- `lang`: Language code (optional, default: "spa")

**Response:**

```json
{
  "comparison": {
    "tesseract": {
      "text": "Texto extraído por Tesseract...",
      "confidence": 85.5,
      "processing_time": "2.34s",
      "words_count": 45,
      "preprocessing": "Applied"
    },
    "paddleocr": {
      "text": "Texto extraído por PaddleOCR...",
      "confidence": 92.1,
      "processing_time": "1.87s",
      "words_count": 47,
      "preprocessing": "None"
    }
  },
  "recommendation": "PaddleOCR"
}
```

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/ocr-comparison" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.jpg" \
  -F "lang=spa"
```

### 6. API Documentation

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

The service supports multiple languages through Tesseract and PaddleOCR. Common language codes:

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
git commit -m "Initial OCR microservice with dual engines"
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

1. **HTTP Request Node (Process Image):**

   - Method: `POST`
   - URL: `https://yourdomain.com/process-image`
   - Response Format: `File`
   - Binary Property: `data` → File Parameter `file`

2. **HTTP Request Node (OCR Comparison):**

   - Method: `POST`
   - URL: `https://yourdomain.com/ocr-comparison`
   - Response Format: `JSON`
   - Binary Property: `data` → File Parameter `file`

3. **Decision Node:**

   - Use the `recommendation` field to choose the best result
   - Connect to your OCR workflow

### Python Client Example

```python
import requests

def compare_ocr_engines(image_path, base_url="http://localhost:8000"):
    """Compare Tesseract vs PaddleOCR results"""

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{base_url}/ocr-comparison", files=files)

    if response.status_code == 200:
        result = response.json()

        print("=== OCR Comparison Results ===")
        print(f"Tesseract Confidence: {result['comparison']['tesseract']['confidence']:.1f}%")
        print(f"PaddleOCR Confidence: {result['comparison']['paddleocr']['confidence']:.1f}%")
        print(f"Recommendation: {result['recommendation']}")

        return result
    else:
        raise Exception(f"Error: {response.status_code}")

def process_with_best_engine(image_path, base_url="http://localhost:8000"):
    """Process image with the recommended OCR engine"""

    # First, compare engines
    comparison = compare_ocr_engines(image_path, base_url)

    # Use the recommended engine
    if comparison['recommendation'] == 'PaddleOCR':
        endpoint = f"{base_url}/ocr-paddle"
    else:
        endpoint = f"{base_url}/ocr-complete"

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(endpoint, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}")

# Usage
result = process_with_best_engine("invoice.jpg")
print(f"Extracted text: {result['text']}")
```

## 🏗️ Architecture

### OCR Engines Comparison

| Feature             | Tesseract   | PaddleOCR     |
| ------------------- | ----------- | ------------- |
| **Preprocessing**   | ✅ Required | ❌ Not needed |
| **Speed**           | ⭐⭐⭐      | ⭐⭐⭐⭐⭐    |
| **Accuracy**        | ⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐    |
| **Spanish Support** | ⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐    |
| **Text Detection**  | ❌ Manual   | ✅ Automatic  |
| **Memory Usage**    | ⭐⭐⭐⭐⭐  | ⭐⭐⭐⭐      |

### Image Processing Pipeline

1. **Deskew**: Automatic rotation correction
2. **Perspective Warp**: Document alignment and correction
3. **Adaptive Thresholding**: Optimal binarization for OCR
4. **Text Extraction**: Tesseract LSTM engine or PaddleOCR

### Technology Stack

- **FastAPI**: High-performance web framework
- **OpenCV**: Computer vision and image processing
- **Tesseract**: OCR engine with LSTM
- **PaddleOCR**: Advanced OCR engine from Baidu
- **Pillow**: Image manipulation
- **Uvicorn**: ASGI server
- **Docker**: Containerization

## 📊 Performance

- **Processing Time**: ~1-5 seconds per image (depending on engine)
- **Memory Usage**: ~300-800MB per container
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Image Formats**: JPEG, PNG, BMP, TIFF
- **OCR Accuracy**: 85-95% depending on image quality

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

3. **PaddleOCR installation issues:**

```bash
pip install paddlepaddle paddleocr
```

4. **Port already in use:**

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
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for advanced OCR capabilities
- [Coolify](https://coolify.io/) for easy deployment

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the troubleshooting section above

---

**Made with ❤️ for efficient OCR processing with dual engines**
