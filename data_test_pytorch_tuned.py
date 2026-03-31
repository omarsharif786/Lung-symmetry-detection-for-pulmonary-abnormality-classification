import random
from itertools import product
from os import listdir

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

matplotlib.use("TkAgg")


# Reproducibility
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

image_size = (150, 150)
left_dir = "D:/Datasets/MontgomerySet/preproc_slice/leftmask/"
right_dir = "D:/Datasets/MontgomerySet/preproc_slice/rightmask/"


def data_gen(left_paths, right_paths):
    x_data = []
    y = []

    # Preserve your original labeling strategy
    count = 0
    for lp, rp in zip(left_paths, right_paths):
        left_img = cv2.resize(cv2.imread(lp), image_size)
        right_img = cv2.resize(cv2.imread(rp), image_size)

        concatenated = np.concatenate((left_img, right_img), axis=1)  # (150, 300, 3)
        x_data.append(concatenated)

        count += 1
        y.append(0 if count < 80 else 1)

    return np.array(x_data), np.array(y)


def create_pairs(x, y):
    x_left = []
    x_right = []

    for item in x:
        left_half, right_half = np.split(item, 2, axis=1)  # split width: 300 -> 150 + 150
        x_left.append(left_half)
        x_right.append(right_half)

    pairs = [[l, r] for l, r in zip(x_left, x_right)]
    labels = y.copy()
    return np.array(pairs), np.array(labels)


class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        left_img = self.pairs[idx][0].astype(np.float32) / 255.0
        right_img = self.pairs[idx][1].astype(np.float32) / 255.0

        left_img = torch.from_numpy(left_img).permute(2, 0, 1)
        right_img = torch.from_numpy(right_img).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return left_img, right_img, label


class BaseCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 18 * 18, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.fc_out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        encoded_l = self.base_model(x1)
        encoded_r = self.base_model(x2)
        l1_distance = torch.abs(encoded_l - encoded_r)
        return self.sigmoid(self.fc_out(l1_distance))


def run_one_config(train_dataset, val_dataset, config, epochs=25):
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = SiameseNetwork(BaseCNN(dropout_rate=config["dropout"])).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for left_imgs, right_imgs, labels in train_loader:
            left_imgs = left_imgs.to(device)
            right_imgs = right_imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(left_imgs, right_imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_losses.append(train_loss / max(1, len(train_loader)))
        train_accs.append(train_correct / max(1, train_total))

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for left_imgs, right_imgs, labels in val_loader:
                left_imgs = left_imgs.to(device)
                right_imgs = right_imgs.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(left_imgs, right_imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_losses.append(val_loss / max(1, len(val_loader)))
        val_accs.append(val_correct / max(1, val_total))

    history = {
        "train_losses": train_losses,
        "train_accuracies": train_accs,
        "val_losses": val_losses,
        "val_accuracies": val_accs,
    }
    return model, history, val_accs[-1]


def plot_roc(pred_scores, y_true):
    fpr, tpr, _ = roc_curve(y_true, pred_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()


def main():
    left = [left_dir + f for f in listdir(left_dir)]
    right = [right_dir + f for f in listdir(right_dir)]

    X, y = data_gen(left, right)
    pairs, labels = create_pairs(X, y)

    x_train, x_test, y_train, y_test = train_test_split(
        pairs,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    y_test_original = y_test.copy()

    train_dataset = SiameseDataset(x_train, y_train)
    val_dataset = SiameseDataset(x_test, y_test)
    test_dataset = SiameseDataset(x_test, y_test)

    search_space = {
        "lr": [1e-3, 5e-4],
        "batch_size": [8, 12],
        "dropout": [0.3, 0.5],
        "weight_decay": [1e-4, 2.5e-4],
    }
    tune_epochs = 25

    best_val_acc = -1.0
    best_config = None
    best_history = None
    best_state_dict = None

    for lr, batch_size, dropout_rate, weight_decay in product(
        search_space["lr"],
        search_space["batch_size"],
        search_space["dropout"],
        search_space["weight_decay"],
    ):
        config = {
            "lr": lr,
            "batch_size": batch_size,
            "dropout": dropout_rate,
            "weight_decay": weight_decay,
        }

        model, history, val_acc = run_one_config(train_dataset, val_dataset, config, epochs=tune_epochs)
        print(
            f"Config lr={lr}, batch_size={batch_size}, dropout={dropout_rate}, "
            f"weight_decay={weight_decay} -> val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config
            best_history = history
            best_state_dict = model.state_dict()

    print(f"Best config: {best_config}, best val_acc={best_val_acc:.4f}")

    # Restore best model
    best_model = SiameseNetwork(BaseCNN(dropout_rate=best_config["dropout"])).to(device)
    best_model.load_state_dict(best_state_dict)
    best_model.eval()

    test_loader = DataLoader(test_dataset, batch_size=best_config["batch_size"], shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for left_imgs, right_imgs, labels in test_loader:
            left_imgs = left_imgs.to(device)
            right_imgs = right_imgs.to(device)
            outputs = best_model(left_imgs, right_imgs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    pred = np.array(all_preds)
    y_test_pred = y_test_original

    threshold = 0.8
    y_final = np.where(pred > threshold, 1, 0).flatten()

    score = pred[0][0] if pred.ndim > 1 else pred[0]
    print(
        "This image is %.2f percent normal and %.2f percent abnormal."
        % (100 * (1 - score), 100 * score)
    )

    plot_roc(pred.flatten(), y_test_pred)

    cm = confusion_matrix(y_test_pred, y_final)
    print("confusion matrix")
    print(cm)
    ConfusionMatrixDisplay(cm, display_labels=["normal", "abnormal"]).plot(cmap=plt.cm.Blues)

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    print(cm_normalized)
    ConfusionMatrixDisplay(cm_normalized, display_labels=["normal", "abnormal"]).plot(cmap=plt.cm.Blues)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(best_history["train_accuracies"])
    plt.plot(best_history["val_accuracies"])
    plt.title("Model Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["training accuracy", "validation accuracy"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(best_history["train_losses"])
    plt.plot(best_history["val_losses"])
    plt.title("Training Loss and Validation Loss")
    plt.ylabel("Cross Entropy")
    plt.xlabel("epoch")
    plt.legend(["Training Loss", "Validation Loss"], loc="upper left")
    plt.tight_layout()
    plt.show()

    report = classification_report(y_test_pred, y_final, target_names=["normal", "abnormal"])
    print(report)


if __name__ == "__main__":
    main()
