import pandas as pd

# 配置参数
control_output_path = "D:/AD_detect/control_labeled.csv"  # 正常人输出路径
ad_output_path = "D:/AD_detect/ad_labeled.csv"            # 患者输出路径
file_paths = [
    "D:/AD_detect/metadata_withPauses.csv"
]

# 读取并合并数据
dfs = []
for path in file_paths:
    try:
        df = pd.read_csv(path, encoding='utf-8')
        dfs.append(df)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='gbk')  # 尝试GBK编码
        dfs.append(df)
    except FileNotFoundError:
        print(f"错误：文件 {path} 不存在")
        exit()

combined_df = pd.concat(dfs, axis=0)

# 分类处理
def process_category(data, category_name, label_value):
    filtered = data[data['category'].str.strip().str.lower() == category_name.lower().strip()]
    filtered = filtered.copy()
    filtered['label'] = label_value
    return filtered[['data', 'label']]  # 按需保留其他列

# 生成数据集
control_df = process_category(combined_df, "Control", 1)
probable_ad_df = process_category(combined_df, "ProbableAD", 0)

# 保存结果（UTF-8带BOM解决Excel乱码）
control_df.to_csv(control_output_path, index=False, encoding='utf-8-sig')
probable_ad_df.to_csv(ad_output_path, index=False, encoding='utf-8-sig')

print(f"生成完成：\n正常人 -> {control_output_path} ({len(control_df)}条)\n患者 -> {ad_output_path} ({len(probable_ad_df)}条)")