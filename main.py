from garden_analyzer import GardenAestheticsAnalyzer
import os

def find_attachment11_dir(base_data_dir):
    """
    智能查找附件11的目录。
    会尝试多种可能的命名方式，并能处理文件夹不存在的情况。
    """
    print("\n--- 正在智能查找附件11的目录 ---")

    # 检查基本数据目录是否存在
    if not os.path.exists(base_data_dir):
        print(f"❌ 严重错误: 基本数据目录 '{base_data_dir}' 不存在！")
        print(f"请确保名为 '{base_data_dir}' 的文件夹与你的 Python 脚本在同一个目录下。")
        print(f"当前工作目录是: {os.getcwd()}")
        return None

    print(f"✅ 基本数据目录 '{base_data_dir}' 已找到。")
    print("  其下的内容为:")

    # 列出基本目录下的所有内容，帮助调试
    found_items = os.listdir(base_data_dir)
    for item in found_items:
        print(f"    - {item}")

    # 尝试多种可能的命名方式
    possible_names = [
        "11. 其他园林平面图",
        "11.其他园林平面图",
        "附件11",
    ]

    for name in possible_names:
        path = os.path.join(base_data_dir, name)
        if os.path.exists(path):
            print(f"✅ 成功匹配到附件11目录: '{path}'")
            return path

    # 如果标准名称找不到，则尝试模糊匹配 (查找任何以 '11' 开头的目录)
    print("\n标准名称未匹配，尝试模糊查找以'11'开头的目录...")
    for item in found_items:
        path = os.path.join(base_data_dir, item)
        if os.path.isdir(path) and item.strip().startswith('11'):
            print(f"✅ 成功模糊匹配到附件11目录: '{path}'")
            return path

    print("❌ 错误: 未能找到附件11的目录。请检查文件夹命名。")
    return None


def main():
    """
    主执行函数，分两步完成题目要求：
    1. 基于10个代表园林进行相似度分析，挖掘共性。
    2. 将模型推广到附件11的所有其他园林，验证广效用。
    """
    base_data_dir = "赛题F江南古典园林美学特征建模附件资料"
    analyzer = GardenAestheticsAnalyzer(data_dir=base_data_dir)

    # =========================================================================
    # 第一部分: 相似度分析 (基于10个代表园林)
    # =========================================================================
    print("="*80)
    print("第一部分：十大代表园林相似度分析与共性规律挖掘")
    print("="*80)

    analyzer.process_all_gardens()

    if analyzer.feature_df is not None and not analyzer.feature_df.empty:
        analyzer.analyze_similarity()
    else:
        print("❌ 特征提取失败，无法进行第一部分分析。请检查文件路径和格式。")
        return

    # =========================================================================
    # 第二部分: 广效用验证 (基于附件11的所有园林)
    # =========================================================================
    print("\n" + "="*80)
    print("第二部分：模型广效用验证")
    print("="*80)

    # 【关键修改】: 使用新的智能查找函数
    new_gardens_dir = find_attachment11_dir(base_data_dir)

    if new_gardens_dir is None:
        print("由于未能找到附件11的目录，广效用验证部分无法继续。")
        return

    try:
        test_gardens_files = [f for f in os.listdir(new_gardens_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    except FileNotFoundError:
        print(f"❌ 错误: 无法访问目录: {new_gardens_dir}")
        return

    if not test_gardens_files:
        print(f"⚠️ 警告: 在目录 '{new_gardens_dir}' 中没有找到任何平面图文件 (.jpg, .png)。")
        return

    print(f"\n在附件11目录中找到了 {len(test_gardens_files)} 个平面图，将进行逐一分析...")
    print(f"文件列表: {test_gardens_files}")

    successful_tests = 0
    for garden_file in test_gardens_files:
        new_garden_path = os.path.join(new_gardens_dir, garden_file)
        analyzer.generalize_to_new_garden(new_garden_path)
        successful_tests += 1

    print(f"\n✅ 广效用验证完成，成功分析了 {successful_tests}/{len(test_gardens_files)} 个新园林。")
    print("\n\n🎉 所有分析任务完成！结果已保存在 'results' 文件夹中。")

if __name__ == "__main__":
    main()
