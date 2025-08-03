import runpod
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import io
import base64
import numpy as np

# --- Global Model Cache ---
# Load the model once and cache it for subsequent runs.
# This is crucial for performance in a serverless environment.
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Loads and caches the model and processor."""
    global model, processor
    if model is None or processor is None:
        model_id = "facebook/sam-vit-large"
        model = SamModel.from_pretrained(model_id).to(device)
        processor = SamProcessor.from_pretrained(model_id)
    return model, processor

# --- The Handler Function ---
# This function will be called by RunPod for each API request.

def handler(job):
    """
    The main handler function for processing segmentation and cutting jobs.
    The 'job' input is a dictionary containing the request payload.
    """
    model, processor = load_model()
    
    job_input = job['input']
    
    # --- Input Validation ---
    if 'action' not in job_input:
        return {"error": "No action specified. Use 'segment' or 'cut'."}
        
    action = job_input['action']
    
    # Decode the base64 image sent from the frontend
    image_data = base64.b64decode(job_input["image_base64"])
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # --- Perform the requested action ---

    if action == 'segment':
        points = job_input['points']
        img_width, img_height = pil_image.size
        
        input_points = [[[p["x"] * img_width, p["y"] * img_height] for p in points]]
        input_labels = [[p["label"] for p in points]]

        inputs = processor(
            pil_image, 
            input_points=input_points, 
            input_labels=input_labels, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        masks = processor.image_processor.post_process_masks(
    outputs.pred_masks,
    inputs["original_sizes"],
    inputs["reshaped_input_sizes"],
)

        scores = outputs.iou_scores[0].cpu().numpy()
        best_mask_idx = np.argmax(scores)

# ✅ Proper fix
        best_mask = masks[0][best_mask_idx].cpu().numpy().astype(np.uint8)
        
        # Return the result as a JSON object
        return {
            "mask": best_mask.flatten().tolist(),
            "scores": scores.tolist(),
            "width": best_mask.shape[1],
            "height": best_mask.shape[0],
        }

    elif action == 'cut':
        mask_data = base64.b64decode(job_input["mask_base64"])
        mask = Image.open(io.BytesIO(mask_data)).convert("L")
        
        rgba_image = pil_image.convert("RGBA")
        
        if rgba_image.size != mask.size:
            mask = mask.resize(rgba_image.size, Image.NEAREST)

        cut_image = Image.new("RGBA", rgba_image.size, (0, 0, 0, 0))
        cut_image.paste(rgba_image, (0, 0), mask)

        # Save the result to a byte buffer and encode as base64
        byte_arr = io.BytesIO()
        cut_image.save(byte_arr, format='PNG')
        encoded_string = base64.b64encode(byte_arr.getvalue()).decode('utf-8')

        # Return the cut image as a base64 string in a JSON object
        return {"image_base64": encoded_string}
        
    else:
        return {"error": f"Invalid action: {action}"}
