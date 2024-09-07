import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class TrainModel:
    def __init__(
        self, 
        model,
        num_classes: int = 3,
        learning_rate: float = 0.0003,
        verbose: bool = False
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f"Using device: {self.device}")

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.verbose = verbose

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
    def __train_epoch(self, dataloader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def __validate(self, dataloader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, num_epochs: int = 10):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.__train_epoch(train_loader, self.criterion, self.optimizer)
            val_loss, val_acc = self.__validate(val_loader, self.criterion)
            
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def show_plot(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label="Train Accuracy")
        plt.plot(self.val_accs, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def model_save(self, model_name: str):
        torch.save(self.model.state_dict(), model_name)
        if self.verbose:
            print(f"Model saved as '{model_name}'")
