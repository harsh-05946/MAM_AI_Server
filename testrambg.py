from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from rembg import remove
from PIL import Image
import io

app = FastAPI()

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    try:
        # Read image bytes
        input_bytes = await file.read()

        # Remove background
        output_bytes = remove(input_bytes)

        # Return as PNG (transparent)
        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="image/png"
        )

    except Exception as e:
        return {"error": str(e)}