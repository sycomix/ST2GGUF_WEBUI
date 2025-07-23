from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from pathlib import Path
import zipfile

from converter import convert_to_gguf, load_safetensors, weighted_average_tensors

app = FastAPI()

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("static/index.html")

async def _handle_uploaded_model(uploaded_file: UploadFile, session_dir: Path):
    """Handles saving and extracting an uploaded model file (safetensors or zip)."""
    file_path = session_dir / uploaded_file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    input_for_converter = file_path

    if uploaded_file.filename.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(session_dir)
        
        # Determine the actual input path for load_safetensors from the extracted content
        if (session_dir / "model.safetensors.index.json").exists():
            input_for_converter = session_dir
        else:
            safetensors_files = list(session_dir.glob("*.safetensors"))
            if len(safetensors_files) == 1:
                input_for_converter = safetensors_files[0]
            elif len(safetensors_files) > 1:
                input_for_converter = session_dir
            else:
                raise HTTPException(status_code=400, detail=f"No .safetensors or index.json found in the uploaded zip file: {uploaded_file.filename}")
    
    return input_for_converter

@app.post("/convert/")
async def convert_safetensors(file: UploadFile = File(...), quantization: str = Form("none")):
    session_dir = Path("temp_uploads") / os.urandom(8).hex()
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        input_for_converter = await _handle_uploaded_model(file, session_dir)

        output_filename = f"{Path(file.filename).stem}.gguf"
        output_file_path = session_dir / output_filename

        tensors = load_safetensors(str(input_for_converter))
        convert_to_gguf(tensors, str(output_file_path), quantization)

        return {"message": "Conversion successful", "output_file": str(output_file_path)}
    except Exception as e:
        shutil.rmtree(session_dir) # Clean up on error
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/merge_and_convert_models/")
async def merge_and_convert_models(
    model1_file: UploadFile = File(...),
    weight1: float = Form(...),
    model2_file: UploadFile = File(...),
    weight2: float = Form(...),
    quantization: str = Form("none")
):
    session_dir = Path("temp_uploads") / os.urandom(8).hex()
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_for_converter1 = await _handle_uploaded_model(model1_file, session_dir)
        input_for_converter2 = await _handle_uploaded_model(model2_file, session_dir)

        tensors1 = load_safetensors(str(input_for_converter1))
        tensors2 = load_safetensors(str(input_for_converter2))

        merged_tensors = weighted_average_tensors(tensors1, tensors2, weight1, weight2)

        output_filename = f"merged_model_{os.urandom(4).hex()}.gguf"
        output_file_path = session_dir / output_filename

        convert_to_gguf(merged_tensors, str(output_file_path), quantization)

        return {"message": "Merge and conversion successful", "output_file": str(output_file_path)}
    except Exception as e:
        shutil.rmtree(session_dir) # Clean up on error
        raise HTTPException(status_code=500, detail=str(e))