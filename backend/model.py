import base64
from io import BytesIO

import numpy as np
import torch

from PIL import Image
from torchvision import models, transforms
from huggingface_hub import hf_hub_download


# =========================================================
# RETINEX MODEL CONFIGURATION
# =========================================================

REPO_ID = "Arko007/Diabetic-Retinopathy"

MODEL_FILE = "best_model_1024px.pth"


DR_LABELS = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative Diabetic Retinopathy",
    2: "Moderate Non-Proliferative Diabetic Retinopathy",
    3: "Severe Non-Proliferative Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy",
}


# =========================================================
# DEVICE
# =========================================================

device = torch.device("cpu")


print("Loading RETINEX AI model...")


# =========================================================
# RESNET50
# =========================================================

model = models.resnet50(
    weights=None
)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    5
)


# =========================================================
# LOAD MODEL
# =========================================================

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=MODEL_FILE
)


print("Model file:")
print(model_path)


checkpoint = torch.load(
    model_path,
    map_location=device
)


model.load_state_dict(
    checkpoint
)

model.to(device)

model.eval()


print(
    "RETINEX model loaded successfully."
)


# =========================================================
# PREPROCESSING
# =========================================================

transform = transforms.Compose([
    transforms.Resize(
        (1024, 1024)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(
    image,
    target_class
):

    activations = []
    gradients = []


    # Final convolutional block
    target_layer = model.layer4[-1]


    # -----------------------------------------------------
    # Forward hook
    # -----------------------------------------------------

    def forward_hook(
        module,
        inputs,
        output
    ):

        activations.append(
            output.detach()
        )


    # -----------------------------------------------------
    # Gradient hook
    # -----------------------------------------------------

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        if (
            grad_output
            and grad_output[0]
            is not None
        ):

            gradients.append(
                grad_output[0].detach()
            )


    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )


    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )


    try:

        image_rgb = image.convert(
            "RGB"
        )


        tensor = transform(
            image_rgb
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            device
        )


        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        model.zero_grad()

        output = model(
            tensor
        )


        target_score = output[
            0,
            target_class
        ]


        # -------------------------------------------------
        # Backward
        # -------------------------------------------------

        target_score.backward()


        if not activations:
            return None

        if not gradients:
            return None


        activation = activations[0]

        gradient = gradients[0]


        # -------------------------------------------------
        # Grad-CAM weights
        # -------------------------------------------------

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )


        cam = (
            weights *
            activation
        ).sum(
            dim=1,
            keepdim=True
        )


        # ReLU
        cam = torch.relu(
            cam
        )


        cam = cam.squeeze(
            0
        ).squeeze(
            0
        )


        cam = cam.cpu().numpy()


        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 0:
            cam /= maximum


        # -------------------------------------------------
        # Convert to image
        # -------------------------------------------------

        cam = (
            cam * 255
        ).astype(
            np.uint8
        )


        heatmap = Image.fromarray(
            cam
        )


        heatmap = heatmap.resize(
            image_rgb.size,
            Image.Resampling.BILINEAR
        )


        # -------------------------------------------------
        # Create color heatmap
        # -------------------------------------------------

        heatmap_array = np.array(
            heatmap
        ).astype(
            np.float32
        ) / 255.0


        # Smooth pseudo-color map
        red = np.clip(
            heatmap_array * 1.5,
            0,
            1
        )

        green = np.clip(
            1.4 *
            (
                1 -
                np.abs(
                    heatmap_array -
                    0.5
                ) * 2
            ),
            0,
            1
        )

        blue = np.clip(
            1 -
            heatmap_array * 1.4,
            0,
            1
        )


        heatmap_rgb = np.stack(
            [
                red,
                green,
                blue
            ],
            axis=2
        )


        heatmap_rgb = (
            heatmap_rgb * 255
        ).astype(
            np.uint8
        )


        heatmap_image = Image.fromarray(
            heatmap_rgb
        )


        # -------------------------------------------------
        # Blend
        # -------------------------------------------------

        output_size = (
            512,
            512
        )


        original_small = (
            image_rgb.resize(
                output_size,
                Image.Resampling.LANCZOS
            )
        )


        heatmap_small = (
            heatmap_image.resize(
                output_size,
                Image.Resampling.BILINEAR
            )
        )


        overlay = Image.blend(
            original_small,
            heatmap_small,
            0.45
        )


        # -------------------------------------------------
        # Encode JPEG
        # -------------------------------------------------

        buffer = BytesIO()

        overlay.save(
            buffer,
            format="JPEG",
            quality=85
        )


        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode(
            "utf-8"
        )


        return (
            "data:image/jpeg;base64,"
            + encoded
        )


    finally:

        forward_handle.remove()

        backward_handle.remove()

        model.zero_grad()


# =========================================================
# PREDICTION
# =========================================================

def predict_image(
    image: Image.Image
):

    image = image.convert(
        "RGB"
    )


    tensor = transform(
        image
    )


    tensor = tensor.unsqueeze(
        0
    )


    tensor = tensor.to(
        device
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = (
            torch.softmax(
                output,
                dim=1
            )
        )


    predicted_class = (
        torch.argmax(
            probabilities,
            dim=1
        ).item()
    )


    confidence = (
        probabilities[
            0,
            predicted_class
        ].item()
    )


    # -----------------------------------------------------
    # Class probabilities
    # -----------------------------------------------------

    class_probabilities = {}

    for i in range(5):

        class_probabilities[
            i
        ] = round(
            probabilities[
                0,
                i
            ].item() * 100,
            2
        )


    # -----------------------------------------------------
    # Grad-CAM
    # -----------------------------------------------------

    print(
        "Generating Grad-CAM..."
    )


    gradcam_image = (
        generate_gradcam(
            image,
            predicted_class
        )
    )


    print(
        "Grad-CAM generated."
    )


    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return {

        "dr_level":
            predicted_class,

        "diagnosis":
            DR_LABELS[
                predicted_class
            ],

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "referable":
            predicted_class >= 2,

        "class_probabilities":
            class_probabilities,

        "gradcam_image":
            gradcam_image,
    }