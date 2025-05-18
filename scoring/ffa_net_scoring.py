import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from models.ffa_net import FFANet
from datasets.dotah_dataset import DotahDataset
from torch.utils.data import DataLoader
from utils.data_utils import load_config

def main():
    # Load configuration
    config = load_config('configs/config_ffa_net.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = FFANet(
        num_groups=config['train']['num_groups'],
        num_blocks=config['train']['num_attention_blocks'],
        hidden_dim=config['train']['hidden_dim'],
        kernel_size=config['train']['kernel_size'],
        remove_global_skip_connection=config['train']['remove_global_skip_connection']
    )
    
    # Load model weights
    model_path = 'trained_models/ffa_net_dotah.pth'  # Update this path
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    output_dir = 'output_images/ffa_net_dehazed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Inference loop
    with torch.no_grad():
        for i, (hazy_img, _) in enumerate(dataloader):
            print(f'Dehazing image {i}')
            hazy_img = hazy_img.to(device)
            
            # Forward pass
            dehazed_img = model(hazy_img)
            
            # Convert to PIL image and save
            dehazed_img = dehazed_img.squeeze(0).cpu()
            dehazed_img = transforms.ToPILImage()(dehazed_img)
            
            # Save the dehazed image
            output_path = os.path.join(output_dir, f'dehazed_{i:04d}.png')
            dehazed_img.save(output_path)
            
            print(f'Processed image {i+1}/{len(dataset)}')

if __name__ == '__main__':
    main()
