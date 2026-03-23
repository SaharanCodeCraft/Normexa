from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from PIL import Image
import base64
import cv2
import traceback

# --- REAL AI IMPORT ---
# We are bringing back your real model
from backend.inference import run_patchcore

app = FastAPI()

# --- ENABLE CORS (Keeps the bridge to Frontend open) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_CATEGORIES = [
    "bottle", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper"
]

@app.get("/")
def home():
    return {"message": "Normexa API is Live 🚀"}


@app.post("/predict")
async def predict(category: str = Form(...), file: UploadFile = File(...)):
    try:
        # 1. Validation
        if category not in VALID_CATEGORIES:
            return {"error": f"Invalid category '{category}'."}

        # 2. Prepare Temp Folder
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", file.filename)

        # 3. Save Uploaded File
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📂 Image saved: {file_path}")

        # 4. Validate Image Integrity
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            return {"error": "Invalid image file"}

        # 5. RUN REAL INFERENCE
        print(f"🧠 Running PatchCore for category: {category}...")
        heatmap, score = run_patchcore(category, file_path)
        
        print(f"✅ Inference complete. Score: {score}")

        # 6. Process Heatmap for Display
        output_filename = f"output_{file.filename}.png"
        output_path = os.path.join("temp", output_filename)
        
        # Save heatmap
        # convert heatmap to color
        heatmap_uint8 = (heatmap * 255).astype("uint8")
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        # read original image
        original = cv2.imread(file_path)
        original = cv2.resize(original, (256, 256))
        
        # overlay heatmap on original
        overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
        
        # save overlay
        cv2.imwrite(output_path, overlay)

        # 7. Convert to Base64 (So Frontend can show it)
        with open(output_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        

        LOW = 0.45
        HIGH = 0.60

        if score < LOW:
            prediction = "Normal"
        elif score < HIGH:
            prediction = "Suspicious"
        else:
            prediction = "Defect"
        # 8. Send Response
        return {
            "score": round(float(score), 4),
            "prediction": prediction,
            "image": encoded_string
        }

    except Exception as e:
        print("🔥 CRITICAL ERROR DURING INFERENCE:")
        traceback.print_exc()
        return {"error": str(e)}