import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
import argparse # argparse is good practice, but wandb.config can replace it

def main():
    # 1. Initialize W&B
    # This will automatically read the parameters from the sweep
    wandb.init()
    
    # 2. Get configuration from W&B
    # This is the magic: wandb.config contains all the params
    # from your sweep.yaml (e.g., config.lr, config.optimizer)
    config = wandb.config
    
    # 3. Setup for reproducibility and device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config.seed)
    
    # 4. Load Data (CIFAR-10)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=config.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=config.batch_size,
                                             shuffle=False, num_workers=2)

    # 5. Define Model (ResNet18)
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.to(device)
    
    # 6. Define Loss and Optimizer from config
    criterion = nn.CrossEntropyLoss()
    
    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Tell wandb to watch the model
    wandb.watch(model, log='all', log_freq=100)

    # 7. Training Loop
    # We'll run for 5 epochs for this simple MVP
    print(f"Starting training with: {config}")
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99: # Log every 100 mini-batches
                wandb.log({"train/loss": running_loss / 100, "epoch": epoch})
                running_loss = 0.0

        # 8. Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch}: Val Accuracy: {val_accuracy:.2f} %')
        # Log validation accuracy to wandb
        wandb.log({"val/accuracy": val_accuracy, "epoch": epoch})

    print('Finished Training')
    wandb.finish()

if __name__ == '__main__':
    main()

