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
                                              shuffle=True, num_workers=7, pin_memory=True,
                                              persistent_workers=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=config.batch_size,
                                             shuffle=False, num_workers=7, pin_memory=True,
                                             persistent_workers=True)

    # 5. Define Model (ResNet18)
    model = torchvision.models.resnext50_32x4d(num_classes=10)
    #model = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.DEFAULT, num_classes=10)
    if torch.cuda.device_count() < 2:
        print(f"Warning: Found {torch.cuda.device_count()} GPU(s). A dual-GPU setup was expected.")
        model.to(device)
    else:
        print(f"Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        # This splits the batch_size across all available GPUs
        #model = nn.DataParallel(model)
        model.to(device)
        print(f"Model moved to {device} and wrapped in nn.DataParallel.")

    
    # 6. Define Loss and Optimizer from config
    criterion = nn.CrossEntropyLoss()
    
    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'muon':
        optimizer = optim.Muon(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=1e-4
        )
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Tell wandb to watch the model
    wandb.watch(model, log='all', log_freq=50)

    # autocast for ampere architecture
    scaler = torch.amp.GradScaler("cuda")

    # 7. Training Loop
    # We'll run for 5 epochs for this simple MVP
    print(f"Starting training with: {config}")
    for epoch in range(15):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            #loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            #optimizer.step()
            scaler.update()
            
            running_loss += loss.item()
            if i % 10 == 9: # Log every 20 mini-batches
                wandb.log({"train/loss": running_loss / 100, "epoch": epoch})
                running_loss = 0.0

        # 8. Validation Loop
        if True:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
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
