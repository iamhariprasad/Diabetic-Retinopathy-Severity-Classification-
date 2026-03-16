import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import APTOSDataset
from model import build_model
from ransac_refinement import get_final_vessel_mask
from utils import set_seed, calculate_classification_metrics
from evaluate import evaluate_model

def train_model(config):
    """
    Main training loop for the combined classical + ViT pipeline.
    config should be a dictionary with hyperparameters.
    """
    set_seed(config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optional: We can pass the `get_final_vessel_mask` to dataset,
    # but since it runs cv2 operations, doing this online during training 
    # will be slow. For a real project, we would precompute the masks and save them.
    # To keep it end-to-end as requested, we pass it here.
    train_dataset = APTOSDataset(
        csv_file=config['train_csv'],
        img_dir=config['train_dir'],
        segmenter_fn=get_final_vessel_mask if config.get('use_vessels', True) else None,
        mask_dir='data/masks' if config.get('use_vessels', True) else None,
        is_train=True
    )
    
    val_dataset = APTOSDataset(
        csv_file=config['val_csv'],
        img_dir=config['val_dir'],
        segmenter_fn=get_final_vessel_mask if config.get('use_vessels', True) else None,
        mask_dir='data/masks' if config.get('use_vessels', True) else None,
        is_train=False
    )
    
    # num_workers=0 because cv2/skimage inside workers can sometimes hang
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    in_chans = 4 if config.get('use_vessels', True) else 3
    model = build_model(num_classes=config['num_classes'], in_chans=in_chans, device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    best_val_acc = 0.0
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{config['epochs']}]")
            loop.set_postfix(loss=train_loss / (batch_idx + 1))
            
        scheduler.step()
        
        # Validation
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
            print("=> Saved new best model")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train DR Classification Model")
    parser.add_argument('--train_csv', type=str, default='data/train.csv', help='Path to training CSV')
    parser.add_argument('--train_dir', type=str, default='data/train_images', help='Path to training images directory')
    parser.add_argument('--val_csv', type=str, default='data/val.csv', help='Path to validation CSV')
    parser.add_argument('--val_dir', type=str, default='data/val_images', help='Path to validation images directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--no_vessels', action='store_true', help='Disable vessel segmentation fusion')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model weights')
    
    args = parser.parse_args()
    
    cfg = {
        'seed': 42,
        'train_csv': args.train_csv,
        'train_dir': args.train_dir,
        'val_csv': args.val_csv,
        'val_dir': args.val_dir,
        'use_vessels': not args.no_vessels,
        'num_classes': 5,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'save_dir': args.save_dir
    }
    
    # Run training
    train_model(cfg)
