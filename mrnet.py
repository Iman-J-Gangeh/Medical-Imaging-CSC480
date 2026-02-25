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
from imblearn.over_sampling import SMOTE
from typing import Tuple, List, Optional

DATA_DIR = './data' 
TASK = 'abnormal'         # options: 'acl', 'meniscus', 'abnormal'
PLANE = 'sagittal'   # options: 'axial', 'coronal', 'sagittal'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-5
EPOCHS = 20
WEIGHT_DECAY = 1e-2
PATIENCE = 3

class MRNetDataset(Dataset):
  def __init__(self, root_dir: str, task: str, plane: str, train: bool = True, transform: Optional[transforms.Compose] = None) -> None:
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

  def __len__(self) -> int:
    return len(self.records)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

class SmoteFeatureDataset(Dataset):
  def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
    self.features = torch.tensor(features, dtype=torch.float32)
    self.labels   = torch.tensor(labels,   dtype=torch.float32)

  def __len__(self) -> int:
    return len(self.labels)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return self.features[idx], self.labels[idx]


class MRNet(nn.Module):
  def __init__(self) -> None:
    super(MRNet, self).__init__()
    # pre-trained ResNet-18
    self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)        
    self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
    
    self.classifier = nn.Sequential(
      nn.Linear(512, 128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(128, 1)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
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

def extract_features(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
  model.eval()
  all_features = []
  all_labels = []
  with torch.no_grad():
    for i, (data, label) in enumerate(loader):
      x = data.squeeze(0).to(DEVICE)           # (slices, 3, H, W)
      feats = model.feature_extractor(x)        # (slices, 512, 1, 1)
      feats = feats.view(feats.size(0), -1)     # (slices, 512)
      pooled = torch.max(feats, 0)[0]           # (512,)
      all_features.append(pooled.cpu().numpy())
      all_labels.append(label.item())
      if i % 50 == 0:
        print(f"  Extracting features: {i}/{len(loader)}")
  return np.array(all_features), np.array(all_labels)


def apply_smote(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  unique, counts = np.unique(labels, return_counts=True)
  print(f"Before SMOTE: {dict(zip(unique.astype(int), counts))}")

  smote = SMOTE(random_state=42)
  features_resampled, labels_resampled = smote.fit_resample(features, labels)

  unique_r, counts_r = np.unique(labels_resampled, return_counts=True)
  print(f"After  SMOTE: {dict(zip(unique_r.astype(int), counts_r))}")
  return features_resampled, labels_resampled


def calculate_metrics(labels: List[float], probs: List[float], preds: List[int], loss: float) -> Tuple[float, float, float, float]:
  try:
    auc = roc_auc_score(labels, probs)
  except ValueError:
    auc = 0.5
    
  f1 = f1_score(labels, preds)
  acc = (np.array(preds) == np.array(labels)).mean()
  return loss, acc, auc, f1

def run_epoch(model: nn.Module, loader: DataLoader, optimizer: Optional[optim.Optimizer], criterion: nn.Module, is_train: bool = True) -> Tuple[float, float, float, float]:
  if is_train:
    model.train()
  else:
    model.eval()

  total_loss = 0.0
  all_labels = []
  all_probs = []
  all_preds = []

  for i, (data, label) in enumerate(loader):
    data, label = data.to(DEVICE), label.to(DEVICE)
    
    if is_train and optimizer is not None:
      optimizer.zero_grad()

    output = model(data)
    loss = criterion(output.view(-1), label.view(-1))
    
    if is_train and optimizer is not None:
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
  return calculate_metrics(all_labels, all_probs, all_preds, avg_loss)

def run_feature_epoch(model: nn.Module, loader: DataLoader, optimizer: Optional[optim.Optimizer], criterion: nn.Module, is_train: bool = True) -> Tuple[float, float, float, float]:
  if is_train:
    model.train()
  else:
    model.eval()

  total_loss = 0.0
  all_labels = []
  all_probs = []
  all_preds = []

  context = torch.enable_grad() if is_train else torch.no_grad()
  with context:
    for features, label in loader:
      features, label = features.to(DEVICE), label.to(DEVICE)

      if is_train and optimizer is not None:
        optimizer.zero_grad()

      output = model.classifier(features)     # (batch, 1)
      loss = criterion(output.view(-1), label.view(-1))

      if is_train and optimizer is not None:
        loss.backward()
        optimizer.step()

      probs = torch.sigmoid(output).detach().cpu().numpy().flatten()
      preds = (probs > 0.5).astype(int)
      all_probs.extend(probs.tolist())
      all_preds.extend(preds.tolist())
      all_labels.extend(label.cpu().numpy().flatten().tolist())
      total_loss += loss.item()

  avg_loss = total_loss / len(loader)
  return calculate_metrics(all_labels, all_probs, all_preds, avg_loss)


def train_loop_smote(model: nn.Module, smote_loader: DataLoader, valid_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> None:
  best_val_auc = 0.0
  patience_counter = 0

  for epoch in range(EPOCHS):
    train_loss, train_acc, train_auc, train_f1 = run_feature_epoch(model, smote_loader, optimizer, criterion, is_train=True)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}")

    with torch.no_grad():
      val_loss, val_acc, val_auc, val_f1 = run_epoch(model, valid_loader, None, criterion, is_train=False)
    print(f"Epoch {epoch+1} | Valid Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}")

    if val_auc > best_val_auc:
      best_val_auc = val_auc
      patience_counter = 0
      torch.save(model.state_dict(), f'mrnet_{TASK}_{PLANE}.pth')
      print(f"Saved Best Model (AUC: {best_val_auc:.4f})")
    else:
      patience_counter += 1
      print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
      print("Early stopping triggered.")
      break


def train_loop(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> None:
  best_val_auc = 0.0
  patience_counter = 0
  
  for epoch in range(EPOCHS):
    train_loss, train_acc, train_auc, train_f1 = run_epoch(model, train_loader, optimizer, criterion, is_train=True)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}")

    with torch.no_grad():
      val_loss, val_acc, val_auc, val_f1 = run_epoch(model, valid_loader, None, criterion, is_train=False)
      print(f"Epoch {epoch+1} | Valid Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}")
    
    if val_auc > best_val_auc:
      best_val_auc = val_auc
      patience_counter = 0
      torch.save(model.state_dict(), f'mrnet_{TASK}_{PLANE}.pth')
      print(f"Saved Best Model (AUC: {best_val_auc:.4f})")
    else:
      patience_counter += 1
      print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
      
    if patience_counter >= PATIENCE:
      print("Early stopping triggered.")
      break

def main() -> None:
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
  train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
  valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

  print("Initializing Model...")
  model = MRNet().to(DEVICE)

  criterion = nn.BCEWithLogitsLoss()

  # --- Phase 1: Extract 512-d features from all training exams ---
  print("\nPhase 1: Extracting features from training set (frozen backbone)...")
  train_features, train_labels = extract_features(model, train_loader)
  print(f"  Extracted {train_features.shape[0]} feature vectors of dim {train_features.shape[1]}")

  # --- Phase 2: Apply SMOTE to balance normal vs abnormal ---
  print("\nPhase 2: Applying SMOTE...")
  features_resampled, labels_resampled = apply_smote(train_features, train_labels)

  smote_dataset = SmoteFeatureDataset(features_resampled, labels_resampled)
  smote_loader  = DataLoader(smote_dataset, batch_size=32, shuffle=True, num_workers=0)

  # Only the classifier head is trained on SMOTE features; backbone stays frozen
  optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

  # --- Phase 3: Train classifier on SMOTE-balanced features ---
  print("\nPhase 3: Training classifier on SMOTE-balanced features...")
  train_loop_smote(model, smote_loader, valid_loader, optimizer, criterion)
  print("Training Complete.")

def test() -> None:
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