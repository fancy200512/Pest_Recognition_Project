import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from thop import profile, clever_format
import time
import numpy as np
from tqdm import tqdm

# 导入写的改进模型和数据加载器
from models.lacm_efficientnet import build_lacm_efficientnet
from utils.data_loader import get_dataloaders

def get_baseline_model(num_classes):
    """加载原始的 EfficientNet-B0 作为基线对比"""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def measure_efficiency(model, device="cpu", input_size=(1, 3, 224, 224), iterations=100):
    """测试模型的参数量、FLOPs和推理延迟"""
    model.to(device)
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # 1. 计算参数量和计算复杂度 (FLOPs)
    # thop默认输出的是MACs(乘加次数)，通常1MAC = 2 FLOPs
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2 
    macs_str, params_str = clever_format([macs, params], "%.3f")
    flops_str, _ = clever_format([flops, params], "%.3f")

    # 2. 测量推理延迟
    print(f"  [Info] 正在 {device.upper()} 上进行预热 (Warm-up)...")
    with torch.no_grad():
        for _ in range(20):  # 预热，消除初始化开销
            _ = model(dummy_input)

    print(f"  [Info] 正在 {device.upper()} 上连续推理 {iterations} 次计算平均耗时...")
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000) # 转换为毫秒(ms)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return params, flops, params_str, flops_str, avg_time, std_time

def evaluate_accuracy(model, val_loader, device):
    """在测试集上评估准确率"""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="  [评估中]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return correct / total

def run_benchmark():
    DATA_DIR = "./Clean_Dataset/Clean_Test"
    BATCH_SIZE = 64
    
    # 获取类别数
    print("="*50)
    print("加载数据集中...")
    _, val_loader, num_classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print(f"数据集类别数: {num_classes}")
    
    models_to_test = {
        "Improved (LACM-EfficientNet)": get_baseline_model(num_classes),
        "Baseline (EfficientNet-B0)": build_lacm_efficientnet(num_classes)
    }

    results = {}

    for name, model in models_to_test.items():
        print("="*50)
        print(f"正在测试模型: {name}")
        
        # 1. 理论性能测试 (参数量与计算量)
       
        params, flops, params_str, flops_str, cpu_time, cpu_std = measure_efficiency(model, device="cpu")
        
       
        acc = "尚未加载训练权重"
        weight_file = ""
        
        if name == "Baseline (EfficientNet-B0)":
            weight_file = "best_baseline_efficientnet.pth"
        elif name == "Improved (LACM-EfficientNet)":
            weight_file = "best_lacm_efficientnet.pth"

        # 如果匹配到了权重文件，就尝试加载并评估
        if weight_file:
            try:
                model.load_state_dict(torch.load(weight_file, map_location="cpu"), strict=False)
                print(f"  [Info] 成功加载 {name} 模型权重，开始评估验证集准确率...")
                acc = evaluate_accuracy(model, val_loader, device="cuda" if torch.cuda.is_available() else "cpu")
            except FileNotFoundError:
                print(f"  [Info] 未找到 {weight_file}，跳过准确率评估。")

        # 将当前模型的成绩录入字典
        results[name] = {
            "Params": params_str,
            "FLOPs": flops_str,
            "CPU Time (ms)": f"{cpu_time:.3f} ± {cpu_std:.3f}",
            "Accuracy": acc if isinstance(acc, str) else f"{acc:.4f}"
        }

    # 打印最终对比表格
    print("\n" + "="*85)
    print(f"{'模型对比验证':^85}")
    print("="*85)
    print(f"{'Model':<30} | {'Params':<8} | {'FLOPs':<8} | {'CPU (ms/img)':<18} | {'Accuracy':<10}")
    print("-" * 85)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['Params']:<8} | {metrics['FLOPs']:<8} | {metrics['CPU Time (ms)']:<18} | {metrics['Accuracy']:<10}")
    print("="*85)

if __name__ == "__main__":
    run_benchmark()
