from flask import Flask, request, jsonify
import pickle
import io
from PIL import Image
from flask.debughelpers import FormDataRoutingRedirect
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from flask_cors import CORS, cross_origin

# Load the model
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
checkpoint_path = "/Users/chandanapamidi/Desktop/VLMR/model_best.pth.tar"
# # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# # model.load_state_dict(checkpoint['state_dict'])


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/")
def hello():
    return "Inference for BLIP2_T2 Fine-tuned model on chest x-rays"

@app.route('/generate_caption', methods=['POST'])
@cross_origin()
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    # Get the image file from the request
    file = request.files['image']
    # Read the image file
    # img_bytes = file.read()
    # Convert the image bytes to PIL Image
    image = Image.open(file).convert("RGB")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    
    inputs = processor(images=image, return_tensors="pt", padding=True)

    # Generate the caption
    output = model.generate(**inputs)
    output = processor.decode(output[0])
    return jsonify({'report': output})

if __name__ == '__main__':
    app.run(debug=True)