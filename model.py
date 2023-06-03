from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

# Define the API endpoint for generating images
@app.route('/generate_images', methods=['POST'])
def generate_images():
    # Get the prompt from the request body
    prompt = request.json['prompt']
    
    # Generate the image using the diffusion model
    image = pipe(prompt).images[0]
    
    # Save the generated image locally (optional)
    image.save("generated_image.png")
    
    # Return the URL of the generated image as a response
    return jsonify({'image_url': 'generated_image.png'})

# Start the Flask server
if __name__ == '__main__':
    app.run()
