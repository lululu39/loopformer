import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from config import LoopFormerForImageClassificationConfig
from model import LoopFormerForImageClassification

def main():
    # Device and hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    learning_rate = 1e-4
    epochs = 20
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Dataset and dataloader
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model configuration
    config = LoopFormerForImageClassificationConfig(
        hidden_size=256,
        num_loops=4,
        num_heads=8,
        intermediate_size=1024,
        num_classes=10,
        image_size=224,
        patch_size=16,
        num_channels=1,
        modules={
            "full_attention": 1,
            "mlp": 1,
            "swish_glu": 1,
            "identity": 1
        }
    )
    
    # Model, loss, optimizer
    model = LoopFormerForImageClassification(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # log model num of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    best_acc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(data)
                loss = criterion(output, target)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            train_acc = 100. * correct / total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_acc:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # Validation progress bar
        val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                
                # Mixed precision validation
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(data)
                    loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update validation progress bar
                val_acc = 100. * correct / total
                val_pbar.set_postfix({'Acc': f'{val_acc:.2f}%'})
        
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"new best accuracy: {best_acc:.2f}%")
        
        # Print epoch summary
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_test_loss:.4f}, Val Acc: {test_acc:.2f}%')

if __name__ == '__main__':
    main()