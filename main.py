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
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import layoutparser as lp

app = FastAPI()

# Inicializar OCR engines (una sola vez al arrancar)
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='es')

# Inicializar DocTR
doctr_predictor = ocr_predictor(pretrained=True)

# Inicializar LayoutParser (comentado por incompatibilidad)
# layout_model = lp.PaddleDetectionLayoutModel(
#     config_path='lp://PubLayNet/ppyolov2_r50vd_dcn_365e',
#     label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
# )
layout_model = None

def auto_rotate_image(img):
    """Detecta y corrige la orientación de la imagen automáticamente"""
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar texto usando Tesseract
    try:
        # Probar diferentes ángulos
        angles = [0, 90, 180, 270]
        best_angle = 0
        best_confidence = 0
        
        for angle in angles:
            # Rotar imagen
            h, w = gray.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC)
            
            # OCR en la imagen rotada
            config = "--oem 1 --psm 0"  # PSM 0 para detección de orientación
            result = pytesseract.image_to_osd(rotated, config=config)
            
            # Extraer confianza del resultado
            confidence = float(result.split('\n')[1].split(':')[1])
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_angle = angle
        
        # Aplicar la mejor rotación
        if best_angle != 0:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
            
        return img, best_angle
        
    except Exception as e:
        # Si falla, usar método alternativo basado en contornos
        return auto_rotate_contour_based(img)

def auto_rotate_contour_based(img):
    """Método alternativo basado en contornos para rotación"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar contornos de texto
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encontrar el contorno más grande (probablemente texto)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcular el ángulo del rectángulo mínimo
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Corregir ángulo
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Aplicar rotación si es significativa
        if abs(angle) > 2:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
            
        return img, angle
    
    return img, 0

def preprocess_image_np(img):
    """Función reutilizable para preprocesar imagen - Versión mejorada"""
    # 0) Auto-rotación inteligente
    img, rotation_angle = auto_rotate_image(img)
    
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
    clean, rotation_angle = preprocess_image_np(img)

    # Enviar imagen como PNG en la respuesta con información de rotación
    is_success, buffer = cv2.imencode('.png', clean)
    if not is_success:
        raise HTTPException(status_code=500, detail="Image encoding failed")
    
    # Agregar headers con información de rotación
    headers = {
        "X-Image-Rotated": "true" if rotation_angle != 0 else "false",
        "X-Rotation-Angle": str(rotation_angle)
    }
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()), 
        media_type="image/png",
        headers=headers
    )

@app.post("/ocr-text")
async def ocr_text(file: UploadFile = File(...), lang: str = "spa"):
    # 1) Leer imagen en memoria
    contents = await file.read()
    pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # 2) Preprocesado (auto-rotación, deskew, warp, binarización)
    processed, rotation_angle = preprocess_image_np(img)

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

    # 2) Preprocesar imagen (con auto-rotación)
    processed, rotation_angle = preprocess_image_np(img)

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

@app.post("/ocr-doctr")
async def ocr_doctr(file: UploadFile = File(...)):
    """OCR usando DocTR (Document Text Recognition)"""
    
    start_time = time.time()
    
    # 1) Leer imagen en memoria
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2) OCR con DocTR
    try:
        # Convertir a PIL Image para DocTR
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        doc = DocumentFile.from_images([pil_img])
        result = doctr_predictor(doc)
        
        # Extraer texto y métricas
        text = ""
        confidence_scores = []
        word_count = 0
        
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text += word.value + " "
                        confidence_scores.append(word.confidence)
                        word_count += 1
        
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "success": True,
            "text": text.strip(),
            "confidence": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
            "words_count": word_count,
            "processing_time": f"{processing_time:.2f} seconds",
            "engine": "DocTR"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DocTR error: {str(e)}")

@app.post("/ocr-layout")
async def ocr_layout(file: UploadFile = File(...)):
    """OCR usando LayoutParser + PaddleOCR (temporalmente deshabilitado)"""
    
    return JSONResponse({
        "success": False,
        "error": "LayoutParser temporarily disabled due to PaddlePaddle compatibility issues",
        "message": "Please use /ocr-paddle or /ocr-doctr instead",
        "available_endpoints": [
            "/ocr-text",
            "/ocr-complete", 
            "/ocr-paddle",
            "/ocr-doctr",
            "/ocr-comparison"
        ]
    })

@app.post("/ocr-comparison")
async def ocr_comparison(file: UploadFile = File(...), lang: str = "spa"):
    """Compara resultados de Tesseract vs PaddleOCR vs DocTR vs LayoutParser"""
    
    # 1) Leer imagen en memoria
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    results = {}
    
    # 2) Tesseract con preprocesado optimizado
    try:
        tesseract_start = time.time()
        processed = preprocess_image_np(img)
        config = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáéíóúñÁÉÍÓÚÑ.,;:()$%&@#!?-_+=/ "
        tesseract_text = pytesseract.image_to_string(processed, lang=lang, config=config)
        tesseract_data = pytesseract.image_to_data(
            processed, lang=lang, config=config, output_type=pytesseract.Output.DICT
        )
        tesseract_time = time.time() - tesseract_start
        tesseract_confidence = float(np.mean(tesseract_data['conf'])) if tesseract_data['conf'] else 0.0
        
        results["tesseract"] = {
            "text": tesseract_text.strip(),
            "confidence": tesseract_confidence,
            "processing_time": f"{tesseract_time:.2f}s",
            "words_count": len([w for w in tesseract_data['text'] if w.strip()]) if tesseract_data['text'] else 0,
            "preprocessing": "Applied",
            "rotation_applied": rotation_angle != 0,
            "rotation_angle": rotation_angle
        }
    except Exception as e:
        results["tesseract"] = {"error": str(e)}

    # 3) PaddleOCR sin preprocesado
    try:
        paddle_start = time.time()
        paddle_results = paddle_ocr.ocr(img)
        paddle_time = time.time() - paddle_start
        
        paddle_text = ""
        paddle_confidence_scores = []
        if paddle_results and paddle_results[0]:
            for line in paddle_results[0]:
                if line and len(line) >= 2:
                    if isinstance(line[1], list) and len(line[1]) >= 2:
                        paddle_text += line[1][0] + " "
                        paddle_confidence_scores.append(line[1][1])
                    elif isinstance(line[1], str):
                        paddle_text += line[1] + " "
                        paddle_confidence_scores.append(0.0)
        
        paddle_confidence = float(np.mean(paddle_confidence_scores)) if paddle_confidence_scores else 0.0
        
        results["paddleocr"] = {
            "text": paddle_text.strip(),
            "confidence": paddle_confidence,
            "processing_time": f"{paddle_time:.2f}s",
            "words_count": len(paddle_confidence_scores),
            "preprocessing": "None"
        }
    except Exception as e:
        results["paddleocr"] = {"error": str(e)}

    # 4) DocTR
    try:
        doctr_start = time.time()
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        doc = DocumentFile.from_images([pil_img])
        doctr_result = doctr_predictor(doc)
        
        doctr_text = ""
        doctr_confidence_scores = []
        doctr_word_count = 0
        
        for page in doctr_result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        doctr_text += word.value + " "
                        doctr_confidence_scores.append(word.confidence)
                        doctr_word_count += 1
        
        doctr_time = time.time() - doctr_start
        doctr_confidence = float(np.mean(doctr_confidence_scores)) if doctr_confidence_scores else 0.0
        
        results["doctr"] = {
            "text": doctr_text.strip(),
            "confidence": doctr_confidence,
            "processing_time": f"{doctr_time:.2f}s",
            "words_count": doctr_word_count,
            "preprocessing": "None"
        }
    except Exception as e:
        results["doctr"] = {"error": str(e)}

    # 5) LayoutParser + PaddleOCR (temporalmente deshabilitado)
    results["layoutparser"] = {
        "text": "LayoutParser temporarily disabled",
        "confidence": 0.0,
        "processing_time": "0.00s",
        "words_count": 0,
        "preprocessing": "Disabled",
        "error": "PaddlePaddle compatibility issue"
    }

    # 6) Determinar recomendación
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_engine = max(valid_results.keys(), key=lambda x: valid_results[x].get("confidence", 0))
        recommendation = best_engine
    else:
        recommendation = "None"

    return JSONResponse({
        "comparison": results,
        "recommendation": recommendation
    })

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 