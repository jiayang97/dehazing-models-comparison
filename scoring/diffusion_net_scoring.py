import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import yaml
from models.diffusion_net import DDPMNet, Diffusion
from datasets.dotah_dataset import DotahDataset
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config('configs/config_diffusion_net.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = DDPMNet(
        in_channel=3,
        out_channel=3,
        hidden_dim=config['train']['hidden_dim'],
        time_emb_dim=config['train']['time_emb_dim'],
        kernel_size=config['train']['kernel_size']
    )
    
    # Create diffusion process
    diffusion = Diffusion(
        timesteps=config['train']['timesteps'],
        beta=config['train']['beta']
    )
    
    # Load model weights
    model_path = 'trained_models/diffusion_net_dotah.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to a fixed size
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    dataset = DotahDataset(
        hazy_dir='datasets/dota_hazed',
        clear_dir='datasets/dota_clear',
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        num_workers=config['train']['num_workers']
    )
    
    # Create output directory for dehazed images
    output_dir = 'output_images/diffusion_net_dehazed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Inference loop
    with torch.no_grad():
        for i, (hazy_img, _) in enumerate(dataloader):
            print(f'Dehazing image {i}')
            hazy_img = hazy_img.to(device)
            
            # Generate dehazed image using diffusion process
            dehazed_img = diffusion.sample(model, hazy_img, device)
            
            # Convert to PIL image and save
            dehazed_img = dehazed_img.squeeze(0).cpu()
            dehazed_img = transforms.ToPILImage()(dehazed_img)
            
            # Save the dehazed image
            output_path = os.path.join(output_dir, f'dehazed_{i:04d}.png')
            dehazed_img.save(output_path)
            
            print(f'Processed image {i+1}/{len(dataset)}')

if __name__ == '__main__':
    main()
