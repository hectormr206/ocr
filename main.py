from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import cv2
import numpy as np
import io
from PIL import Image
import pytesseract

app = FastAPI()

def preprocess_image_np(img):
    """Función reutilizable para preprocesar imagen"""
    # 1) Deskew (rotación automática)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 200))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskew = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    # 2) Perspective warp (si detecta un contorno cuadrado)
    gray2 = cv2.cvtColor(deskew, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray2, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = np.float32([c[0] for c in approx])
            (tl, tr, br, bl) = pts[np.argsort(pts[:,1])]
            width = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
            height = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
            dst = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
            M2 = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(deskew, M2, (width, height))
        else:
            warped = deskew
    else:
        warped = deskew

    # 3) Binarización optimizada para OCR
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Binarización adaptativa con parámetros optimizados
    clean = cv2.adaptiveThreshold(warped_gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 25, 10)
    
    # Limpieza sutil con morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    
    return clean

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

    # 3) OCR con Tesseract LSTM (--oem 1) y PSM automático (--psm 3)
    config = "--oem 1 --psm 3"
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

    # 3) OCR con Tesseract
    config = "--oem 1 --psm 3"
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

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 