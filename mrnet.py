import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import roc_auc_score, f1_score

DATA_DIR = './' 
TASK = 'abnormal'         # options: 'acl', 'meniscus', 'abnormal'
PLANE = 'sagittal'   # options: 'axial', 'coronal', 'sagittal'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-5
EPOCHS = 1


class MRNetDataset(Dataset):
    def __init__(self, root_dir, task, plane, train=True, transform=None):
        self.root_dir = root_dir
        self.task = task
        self.plane = plane
        self.transform = transform
        self.train = train
        
        # determine folder based on train/valid split
        if self.train:
            self.folder_path = os.path.join(self.root_dir, 'train', plane)
            csv_path = os.path.join(self.root_dir, f'train-{task}.csv')
        else:
            self.folder_path = os.path.join(self.root_dir, 'valid', plane)
            csv_path = os.path.join(self.root_dir, f'valid-{task}.csv')
            
        self.records = pd.read_csv(csv_path, header=None, names=['id', 'label'])
        
        self.records['id'] = self.records['id'].apply(lambda x: str(x).zfill(4)) # Ensure 0000 format

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        case_id = self.records.iloc[idx]['id']
        label = self.records.iloc[idx]['label']
        
        # Load .npy file (slices, 256, 256)
        file_path = os.path.join(self.folder_path, f"{case_id}.npy")
        series = np.load(file_path)
        
        # convert to tensor
        # .npy is (depth, height, width). PyTorch CNNs expect 3 channels (RGB).
        # We stack the grayscale image 3 times to satisfy pre-trained models.
        series = torch.tensor(series, dtype=torch.float32)
        series = torch.stack((series,)*3, axis=1)

        if self.transform:
            series = torch.stack([self.transform(slice_) for slice_ in series])

        label = torch.tensor(float(label), dtype=torch.float32)
        return series, label


class MRNet(nn.Module):
    def __init__(self):
        super(MRNet, self).__init__()
        # pre-trained ResNet-18
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)        
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        # squeeze batch dim to get: (slices, 3, 256, 256)
        x = x.squeeze(0) 
        
        # pass all slices through feature extractor
        # output: (slices, 512, 1, 1)
        features = self.feature_extractor(x)
        
        # flatten: (slices, 512)
        features = features.view(features.size(0), -1)
        
        # MAX POOLING Aggregation
        # take the max value across slices for each feature to find abnormalities
        # Shape: (1, 512)
        pooled_features = torch.max(features, 0, keepdim=True)[0]
        
        output = self.classifier(pooled_features)
        return output


def run_epoch(model, loader, optimizer, criterion, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_labels = []
    all_probs = []
    all_preds = []

    for i, (data, label) in enumerate(loader):
        data, label = data.to(DEVICE), label.to(DEVICE)
        
        if is_train:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output.view(-1), label.view(-1))
        
        if is_train:
            loss.backward()
            optimizer.step()

        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0      
        
        all_probs.append(prob)
        all_preds.append(pred)
        all_labels.append(label.item())
        
        total_loss += loss.item()

        if i % 10 == 0: print(f"{'Train' if is_train else 'Valid'} Step {i}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    
    auc = roc_auc_score(all_labels, all_probs)
    
    f1 = f1_score(all_labels, all_preds)
    
    acc = (np.array(all_preds) == np.array(all_labels)).mean()

    return avg_loss, acc, auc, f1


def main():
    # ImageNet Normalization (Required for pre-trained models)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), # ResNet expects 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Initializing Dataset for Task: {TASK}, Plane: {PLANE}...")
    train_dataset = MRNetDataset(DATA_DIR, TASK, PLANE, train=True, transform=transform)
    valid_dataset = MRNetDataset(DATA_DIR, TASK, PLANE, train=False, transform=transform)

    # batch size 1 because slice depth varies between patients
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Initializing Model...")
    model = MRNet().to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting Training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_auc, train_f1 = run_epoch(model, train_loader, optimizer, criterion, is_train=True)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}")

        with torch.no_grad():
            val_loss, val_acc, val_auc, val_f1 = run_epoch(model, valid_loader, optimizer, criterion, is_train=False)
            print(f"Epoch {epoch+1} | Valid Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}")

            
    print("Training Complete.")
    
    torch.save(model.state_dict(), f'mrnet_{TASK}_{PLANE}.pth')


def test():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = MRNet().to(DEVICE)
    model.load_state_dict(torch.load(f'mrnet_{TASK}_{PLANE}.pth', map_location=DEVICE))

    valid_dataset = MRNetDataset(DATA_DIR, TASK, PLANE, train=False, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    val_loss, val_acc, val_auc, val_f1 = run_epoch(model, valid_loader, None, criterion, is_train=False)
    
    print(f"Validation Results: Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}, F1={val_f1:.4f}")

if __name__ == "__main__":
    main()
