from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

from io import BytesIO

from model import predict_image


# ---------------------------------------------------------
# FASTAPI
# ---------------------------------------------------------

app = FastAPI(
    title="RuralAI Retina API",
    description="AI-assisted diabetic retinopathy screening",
    version="1.0"
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "RuralAI Retina API is running"
    }


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "ResNet50 DR classifier"
    }


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Check image
    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="Invalid file."
        )


    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )


    # Read file
    contents = await file.read()


    # Maximum 10 MB
    if len(contents) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=400,
            detail="Image must be smaller than 10 MB."
        )


    # Open image
    try:

        image = Image.open(
            BytesIO(contents)
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read image."
        )


    # Run AI
    try:

        result = predict_image(
            image
        )

    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail="AI prediction failed."
        )


    return {
        "success": True,

        "filename":
            file.filename,

        **result
    }