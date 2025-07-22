from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from pathlib import Path

from converter import convert_to_gguf, load_safetensors

app = FastAPI()

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("static/index.html")

@app.post("/convert/")
async def convert_safetensors(file: UploadFile = File(...), quantization: str = "none"):
    try:
        # Create a temporary directory to save the uploaded file
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        input_file_path = temp_dir / file.filename
        with open(input_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Determine the actual input path for load_safetensors
        # If an index.json is uploaded, the input path is the directory
        # Otherwise, it's the single uploaded .safetensors file
        if file.filename == "model.safetensors.index.json":
            input_for_converter = temp_dir # Pass the directory
        else:
            input_for_converter = input_file_path # Pass the file

        output_filename = f"{input_file_path.stem}.gguf"
        output_file_path = temp_dir / output_filename

        # Perform the conversion
        tensors = load_safetensors(str(input_for_converter))
        convert_to_gguf(tensors, str(output_file_path), quantization)

        return {"message": "Conversion successful", "output_file": str(output_file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))