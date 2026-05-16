import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

# 导入你的数据加载器和刚才修复好的模型
from utils.data_loader import get_dataloaders
from models.lacm_efficientnet import build_lacm_efficientnet

def train():
    DATA_DIR = "./Clean_Dataset/Clean_Train" 
    
    BATCH_SIZE = 64  
    EPOCHS = 85           # 总周期数
    FREEZE_EPOCHS = 5     # 冻结老权重，只让新模块适应前 5 个 Epoch
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Using device: {DEVICE}")

    # 准备数据
    train_loader, val_loader, num_classes = get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    print(f"Total classes detected: {num_classes}")

    # 初始化模型
    model = build_lacm_efficientnet(num_classes).to(DEVICE)
    print("[Info] 启动两阶段微调：当前冻结 EfficientNet 主干，仅训练 LACM 与分类头...")
    for name, param in model.named_parameters():
        if "spatial_conv" in name or "channel_fc" in name or "classifier" in name:
            param.requires_grad = True  # 新加入的模块，允许更新
        else:
            param.requires_grad = False # 原版老模块，冻结保护
    # ======================================================================
    
    # 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        if epoch == FREEZE_EPOCHS:
            print("\n🔓 [Info] 预热结束！解冻全部主干网络开始训练！")
            for param in model.parameters():
                param.requires_grad = True
        # ======================================================================

        # ---------------- 训练阶段 ----------------
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
            
            # 实时更新进度条显示的指标
            train_bar.set_postfix({'Loss': f'{running_loss/total:.4f}', 'Acc': f'{correct/total:.4f}'})
            
        scheduler.step()

        # ---------------- 验证阶段 ----------------
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

        # ---------------- 保存最优模型 ----------------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_lacm_efficientnet.pth")
            print(f"🌟 New Improved best model saved with accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()