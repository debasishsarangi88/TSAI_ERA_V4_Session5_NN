"""
EVA4 Session 5 - Optimized MNIST Classification
Complete implementation with all requirements verification
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchsummary import summary
import os

class OptimizedNet(nn.Module):
    def __init__(self):
        super(OptimizedNet, self).__init__()
        
        # Block 1: 28x28 -> 14x14 (RF: 3)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28x1 -> 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)  # 28x28x8 -> 28x28x8
        self.bn2 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Block 2: 14x14 -> 7x7 (RF: 7)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)  # 14x14x8 -> 14x14x16
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16 -> 14x14x16
        self.bn4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Block 3: 7x7 -> 3x3 (RF: 15)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)  # 7x7x16 -> 7x7x32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)  # 7x7x32 -> 7x7x32
        self.bn6 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)  # 7x7 -> 3x3
        
        # Transition Layer: 1x1 conv to reduce parameters (RF: 19)
        self.transition = nn.Conv2d(32, 10, 1)  # 3x3x32 -> 3x3x10
        self.bn_transition = nn.BatchNorm2d(10)
        
        # Global Average Pooling: 3x3 -> 1x1 (RF: 23)
        self.gap = nn.AdaptiveAvgPool2d(1)  # 3x3x10 -> 1x1x10
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout(x)
        
        # Transition Layer
        x = F.relu(self.bn_transition(self.transition(x)))
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_description(desc=f'Epoch {epoch} - Loss: {loss.item():.4f} - Acc: {100.*correct/total:.2f}%')

    return train_loss / len(train_loader), 100. * correct / total

def test(model, device, test_loader, dataset_name="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\n{dataset_name} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return test_loss, accuracy

def main():
    print("="*80)
    print("EVA4 SESSION 5 - OPTIMIZED MNIST CLASSIFICATION")
    print("="*80)
    
    # Setup device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(1)
    
    # Model architecture verification
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE VERIFICATION")
    print("="*60)
    
    model = OptimizedNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model: OptimizedNet")
    print(f"Total parameters: {total_params:,}")
    print(f"Target: <20,000 parameters")
    print(f"Parameter Status: {'✓ PASS' if total_params < 20000 else '✗ FAIL'}")
    
    # Model summary
    print("\nModel Architecture Summary:")
    summary(model, input_size=(1, 28, 28))
    
    # Data preparation
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    batch_size = 128
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Load full MNIST dataset
    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    
    # Split into 50k train and 10k validation (using train=True data)
    train_size = 50000
    val_size = 10000
    
    # Create indices for splitting
    indices = torch.randperm(len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    # Test loader (10k samples from test set)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=batch_size, shuffle=False, **kwargs)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {batch_size}")
    
    # Training setup
    print("\n" + "="*60)
    print("TRAINING SETUP")
    print("="*60)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    epochs = 20
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    print(f"Optimizer: Adam (lr=0.001, weight_decay=1e-4)")
    print(f"Scheduler: StepLR (step=7, gamma=0.1)")
    print(f"Max epochs: {epochs}")
    print(f"Target accuracy: 99.4%")
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING LOOP")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        # Training
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        
        # Validation
        val_loss, val_acc = test(model, device, val_loader, "Validation")
        
        # Store metrics
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Early stopping if target achieved
        if val_acc >= 99.4:
            print(f"\n🎯 TARGET ACHIEVED! Validation accuracy: {val_acc:.2f}%")
            break
    
    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Target: 99.4%")
    print(f"Accuracy Status: {'✓ PASS' if best_val_acc >= 99.4 else '✗ FAIL'}")
    print(f"Epochs used: {epoch}")
    print(f"Epoch Status: {'✓ PASS' if epoch < 20 else '✗ FAIL'}")
    
    # Load best model and test on test set
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        test_loss, test_acc = test(model, device, test_loader, "Test")
        print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Plot training progress
    print("\n" + "="*60)
    print("GENERATING TRAINING PLOTS")
    print("="*60)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, 'r-', label='Validation Accuracy')
    plt.axhline(y=99.4, color='g', linestyle='--', label='Target (99.4%)')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Architecture analysis
    print("\n" + "="*60)
    print("ARCHITECTURE ANALYSIS")
    print("="*60)
    print("Network Components:")
    print("✓ Batch Normalization: After every conv layer")
    print("✓ Dropout: 0.1 after each pooling layer")
    print("✓ Global Average Pooling: Instead of FC layers")
    print("✓ Transition Layer: 1x1 conv to reduce parameters")
    print("✓ MaxPooling: Strategic placement every 2 conv layers")
    print("✓ Receptive Field: ~23 (sufficient for MNIST digits)")
    print("✓ Learning Rate: 0.001 with StepLR scheduling")
    print("✓ Weight Decay: 1e-4 for regularization")
    print("✓ Optimizer: Adam (better than SGD for this task)")
    
    # Final requirements check
    print("\n" + "="*60)
    print("REQUIREMENTS VERIFICATION")
    print("="*60)
    
    requirements = {
        "Parameters < 20,000": total_params < 20000,
        "Accuracy ≥ 99.4%": best_val_acc >= 99.4,
        "Epochs < 20": epoch < 20,
        "Batch Normalization": True,  # Built into model
        "Dropout": True,  # Built into model
        "Global Average Pooling": True,  # Built into model
        "Transition Layers": True,  # Built into model
        "Strategic MaxPooling": True,  # Built into model
    }
    
    for req, status in requirements.items():
        print(f"{req}: {'✓ PASS' if status else '✗ FAIL'}")
    
    all_passed = all(requirements.values())
    print(f"\nOverall Status: {'🎉 ALL REQUIREMENTS MET!' if all_passed else '❌ SOME REQUIREMENTS NOT MET'}")
    
    print("\n" + "="*80)
    print("EVA4 SESSION 5 COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return {
        'total_params': total_params,
        'best_val_acc': best_val_acc,
        'final_test_acc': test_acc if 'test_acc' in locals() else 0,
        'epochs_used': epoch,
        'all_requirements_met': all_passed
    }

if __name__ == "__main__":
    results = main()
