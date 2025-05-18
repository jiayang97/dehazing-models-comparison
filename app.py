from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import io
import os
from models.ffa_net import FFANet
from models.diffusion_net import DDPMNet, Diffusion
import yaml

app = Flask(__name__, static_folder='static')

# Load configurations
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Initialize models
def init_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # FFA-Net
    ffa_config = load_config('configs/config_ffa_net.yaml')
    ffa_model = FFANet(
        num_groups=ffa_config['train']['num_groups'],
        num_blocks=ffa_config['train']['num_attention_blocks'],
        hidden_dim=ffa_config['train']['hidden_dim'],
        kernel_size=ffa_config['train']['kernel_size'],
        remove_global_skip_connection=ffa_config['train']['remove_global_skip_connection']
    )
    ffa_model.load_state_dict(torch.load('trained_models/ffa_net_dotah.pth', map_location=device))
    ffa_model.to(device)
    ffa_model.eval()
    
    # Diffusion Net
    diff_config = load_config('configs/config_diffusion_net.yaml')
    diff_model = DDPMNet(
        in_channel=3,
        out_channel=3,
        hidden_dim=diff_config['train']['hidden_dim'],
        time_emb_dim=diff_config['train']['time_emb_dim'],
        kernel_size=diff_config['train']['kernel_size']
    )
    checkpoint = torch.load('trained_models/diffusion_net_dotah.pth', map_location=device)
    diff_model.load_state_dict(checkpoint['model_state_dict'])
    diff_model.to(device)
    diff_model.eval()
    
    diffusion = Diffusion(
        timesteps=diff_config['train']['timesteps'],
        beta=diff_config['train']['beta']
    )
    
    return ffa_model, diff_model, diffusion, device

# Initialize models globally
ffa_model, diff_model, diffusion, device = init_models()

# Transform for images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Add this new route to serve images directly from datasets folder
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('datasets/dota_hazed', filename)

@app.route('/')
def home():
    # Get list of hazy images
    hazy_images = [f for f in os.listdir('datasets/dota_hazed') if f.endswith('.png')]
    return render_template('index.html', hazy_images=hazy_images)

@app.route('/dehaze', methods=['POST'])
def dehaze():
    image_name = request.form['image_name']
    model_type = request.form['model_type']
    
    # Load and transform image
    image_path = os.path.join('datasets/dota_hazed', image_name)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Dehaze using selected model
    with torch.no_grad():
        if model_type == 'ffa':
            output = ffa_model(image_tensor)
        else:  # diffusion
            output = diffusion.sample(diff_model, image_tensor, device)
    
    # Convert output to image
    output = output.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(output)
    
    # Save to bytes
    img_io = io.BytesIO()
    output_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
