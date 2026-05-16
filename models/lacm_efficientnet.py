import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.ops.misc import SqueezeExcitation as SE_Torchvision

class LACM(nn.Module):
    """
    Locally Aware Channel Hybrid Module (LACM)
    """
    def __init__(self, in_channels, reduction=24):
        super(LACM, self).__init__()
        
        # 动态局部核单元（空间分支）深度可分离卷积
        self.spatial_conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, padding=1, 
            groups=in_channels, bias=False
        )
        
        # 通道分组融合单元（通道分支）
        reduced_channels = max(1, in_channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_feat = self.spatial_conv(x)
        channel_feat = self.channel_fc(self.pool(x))
        return spatial_feat * channel_feat


def replace_se_with_lacm(module, depth_level=0):
    """
    递归替换中间和高层特征的 SE 模块为 LACM 模块
    """
    for name, child in module.named_children():
        if isinstance(child, SE_Torchvision):
            # 只替换深层的 SE 模块（浅层不替换）
            if depth_level > 2:
                in_channels = child.fc1.in_channels
                setattr(module, name, LACM(in_channels=in_channels, reduction=24))
        else:
            replace_se_with_lacm(child, depth_level + 1)


def build_lacm_efficientnet(num_classes=10):
    """
    构建带 LACM 改进的 EfficientNet-B0
    """
    # 加载预训练权重
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # 替换 SE 模块
    replace_se_with_lacm(model.features)
    
    # 替换分类头
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

# 测试构建
if __name__ == "__main__":
    model = build_lacm_efficientnet(num_classes=58)
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print("输出形状:", out.shape)
    print("模型构建成功")