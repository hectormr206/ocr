from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import cv2
import numpy as np
import io
from PIL import Image
import pytesseract
from paddleocr import PaddleOCR
import time

app = FastAPI()

# Inicializar PaddleOCR (una sola vez al arrancar)
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='es')

def preprocess_image_np(img):
    """Función reutilizable para preprocesar imagen - Versión mejorada"""
    # 1) Deskew (rotación automática) - Solo si es necesario
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 200))
    
    if len(coords) > 100:  # Solo si hay suficiente contenido
        angle = cv2.minAreaRect(coords)[-1]
        if abs(angle) > 2:  # Solo rotar si el ángulo es significativo
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    # 2) Perspective warp - Solo si detecta documento rectangular
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray2, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        img_area = img.shape[0] * img.shape[1]
        
        # Solo aplicar warp si el contorno es significativo
        if area > img_area * 0.3:  # Al menos 30% del área
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                pts = np.float32([c[0] for c in approx])
                (tl, tr, br, bl) = pts[np.argsort(pts[:,1])]
                width = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
                height = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
                dst = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
                M2 = cv2.getPerspectiveTransform(pts, dst)
                img = cv2.warpPerspective(img, M2, (width, height))

    # 3) Mejora de contraste sin binarización agresiva
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Reducción de ruido suave
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Binarización más suave
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    # Leer imagen en memoria
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocesar imagen
    clean = preprocess_image_np(img)

    # Enviar imagen como PNG en la respuesta
    is_success, buffer = cv2.imencode('.png', clean)
    if not is_success:
        raise HTTPException(status_code=500, detail="Image encoding failed")
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

@app.post("/ocr-text")
async def ocr_text(file: UploadFile = File(...), lang: str = "spa"):
    # 1) Leer imagen en memoria
    contents = await file.read()
    pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # 2) Preprocesado (deskew, warp, binarización)
    processed = preprocess_image_np(img)

    # 3) OCR con Tesseract optimizado para documentos
    config = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáéíóúñÁÉÍÓÚÑ.,;:()$%&@#!?-_+=/ "
    text = pytesseract.image_to_string(processed, lang=lang, config=config)
    data = pytesseract.image_to_data(
        processed, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    # 4) Devolver JSON con texto completo y datos por palabra
    return JSONResponse({
        "text": text,
        "data": data,
        "confidence": float(np.mean(data['conf'])) if data['conf'] else 0.0,
        "words_count": len(data['text']) if data['text'] else 0,
        "language": lang
    })

@app.post("/ocr-complete")
async def ocr_complete(file: UploadFile = File(...), lang: str = "spa"):
    """Procesa imagen y extrae texto en un solo paso (similar a OCR.space)"""
    
    # 1) Leer imagen en memoria
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2) Preprocesar imagen
    processed = preprocess_image_np(img)

    # 3) OCR con Tesseract optimizado para documentos
    config = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáéíóúñÁÉÍÓÚÑ.,;:()$%&@#!?-_+=/ "
    text = pytesseract.image_to_string(processed, lang=lang, config=config)
    data = pytesseract.image_to_data(
        processed, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    # 4) Devolver resultado completo
    return JSONResponse({
        "success": True,
        "text": text.strip(),
        "confidence": float(np.mean(data['conf'])) if data['conf'] else 0.0,
        "words_count": len([w for w in data['text'] if w.strip()]) if data['text'] else 0,
        "language": lang,
        "processing_time": "~2-5 seconds"
    })

@app.post("/ocr-paddle")
async def ocr_paddle(file: UploadFile = File(...)):
    """OCR usando PaddleOCR (con preprocesado opcional)"""
    
    start_time = time.time()
    
    # 1) Leer imagen en memoria
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2) OCR con PaddleOCR
    results = paddle_ocr.ocr(img)
    
    # 3) Extraer texto y métricas
    text = ""
    confidence_scores = []
    word_count = 0
    
    if results and results[0]:
        for line in results[0]:
            if line and len(line) >= 2:
                # line[1] contiene [texto, confianza]
                if isinstance(line[1], list) and len(line[1]) >= 2:
                    text += line[1][0] + " "
                    confidence_scores.append(line[1][1])
                    word_count += 1
                elif isinstance(line[1], str):
                    # Si solo hay texto sin confianza
                    text += line[1] + " "
                    confidence_scores.append(0.0)
                    word_count += 1
    
    processing_time = time.time() - start_time
    
    return JSONResponse({
        "success": True,
        "text": text.strip(),
        "confidence": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
        "words_count": word_count,
        "processing_time": f"{processing_time:.2f} seconds",
        "engine": "PaddleOCR"
    })

@app.post("/ocr-comparison")
async def ocr_comparison(file: UploadFile = File(...), lang: str = "spa"):
    """Compara resultados de Tesseract vs PaddleOCR"""
    
    # 1) Leer imagen en memoria
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2) Tesseract con preprocesado optimizado
    tesseract_start = time.time()
    processed = preprocess_image_np(img)
    config = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáéíóúñÁÉÍÓÚÑ.,;:()$%&@#!?-_+=/ "
    tesseract_text = pytesseract.image_to_string(processed, lang=lang, config=config)
    tesseract_data = pytesseract.image_to_data(
        processed, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    tesseract_time = time.time() - tesseract_start
    tesseract_confidence = float(np.mean(tesseract_data['conf'])) if tesseract_data['conf'] else 0.0

    # 3) PaddleOCR sin preprocesado
    paddle_start = time.time()
    paddle_results = paddle_ocr.ocr(img)
    paddle_time = time.time() - paddle_start
    
    paddle_text = ""
    paddle_confidence_scores = []
    if paddle_results and paddle_results[0]:
        for line in paddle_results[0]:
            if line and len(line) >= 2:
                # line[1] contiene [texto, confianza]
                if isinstance(line[1], list) and len(line[1]) >= 2:
                    paddle_text += line[1][0] + " "
                    paddle_confidence_scores.append(line[1][1])
                elif isinstance(line[1], str):
                    # Si solo hay texto sin confianza
                    paddle_text += line[1] + " "
                    paddle_confidence_scores.append(0.0)
    
    paddle_confidence = float(np.mean(paddle_confidence_scores)) if paddle_confidence_scores else 0.0

    return JSONResponse({
        "comparison": {
            "tesseract": {
                "text": tesseract_text.strip(),
                "confidence": tesseract_confidence,
                "processing_time": f"{tesseract_time:.2f}s",
                "words_count": len([w for w in tesseract_data['text'] if w.strip()]) if tesseract_data['text'] else 0,
                "preprocessing": "Applied"
            },
            "paddleocr": {
                "text": paddle_text.strip(),
                "confidence": paddle_confidence,
                "processing_time": f"{paddle_time:.2f}s",
                "words_count": len(paddle_confidence_scores),
                "preprocessing": "None"
            }
        },
        "recommendation": "PaddleOCR" if paddle_confidence > tesseract_confidence else "Tesseract"
    })

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 