import os

def check_data_leakage(train_dir, test_dir):
    print("="*50)
    print("🔍 启动数据安全审查 (防止数据泄露)...")
    print("="*50)
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("❌ 路径错误，请检查你的文件夹路径是否正确！")
        return

    # 1. 收集训练集所有图片的文件名
    train_files = set()
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_files.add(f)
                
    # 2. 收集测试集所有图片的文件名
    test_files = set()
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_files.add(f)

    # 3. 求交集：检测是否发生数据泄露
    overlap = train_files.intersection(test_files)
    
    print(f"📊 扫描报告：")
    print(f"  - 训练集 (Train) 发现图片: {len(train_files)} 张")
    print(f"  - 测试集 (Test)  发现图片: {len(test_files)} 张")
    print("-" * 50)
    
    if len(overlap) == 0:
        print("✅ Train 和 Test 之间没有任何重叠图片。")
    else:
        print(f"🚨 警告！发生了严重的数据泄露！发现 {len(overlap)} 张图片同时存在于训练集和测试集中！")
        print(f"🚨 泄露的图片示例: {list(overlap)[:5]} ...")

if __name__ == "__main__":
    TRAIN_DIR = "./dataset-second-unzip/Train"  
    TEST_DIR = "./dataset-second-unzip/Test"
    
    check_data_leakage(TRAIN_DIR, TEST_DIR)