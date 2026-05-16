import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# 自定义数据加载器
def get_dataloaders(data_dir, batch_size=32, train_ratio=0.8):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到指定的路径 {data_dir}。")

    print(f"  [Info] 正在从 {data_dir} 读取并划分数据集...")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    num_classes = len(full_dataset.classes)

    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, num_classes


# 训练baseline
def train_baseline():
    # 训练数据集路径
    DATA_DIR = "./Clean_Dataset/Clean_Train" 
    
    BATCH_SIZE = 64  # 如果显存不够（OOM），就调小为 32 或 16
    EPOCHS = 80      # 训练周期
    LEARNING_RATE = 0.001 # 学习率
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")
    # if DEVICE.type == 'cpu':
    #     print("⚠️ 警告: 当前正在使用CPU进行训练")

    # get_dataloaders()函数准备加载数据信息
    train_loader, val_loader, num_classes = get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    print(f"Total classes detected: {num_classes}")

    print("🚀 正在构建Baseline (EfficientNet-B0) 模型...")
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    
    # 官方的b0模型最后全连接层默认是1000分类，这里修改为我论文的实际病害类别数
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(DEVICE)

    # 定义损失函数与优化器：使用Adam优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 初始化最优准确率
    best_acc = 0.0

    # 遍历每一次epoch
    for epoch in range(EPOCHS):
        # 开始训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) 
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 实时更新进度条显示指标：损失，准确率
            train_bar.set_postfix({'Loss': f'{running_loss/total:.4f}', 'Acc': f'{correct/total:.4f}'})
            
        scheduler.step()

        # 开始验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0) 
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = val_correct / val_total
        val_avg_loss = val_loss / val_total
        print(f"Validation Accuracy: {val_acc:.4f} | Validation Loss: {val_avg_loss:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            # 保留baseline的最有权重文件
            torch.save(model.state_dict(), "best_baseline_efficientnet.pth")
            print(f"🌟 New Baseline best model saved with accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_baseline()