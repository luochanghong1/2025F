
import os
import cv2
import numpy as np
import pandas as pd
from skimage import measure
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm
import warnings

# --- 配置 ---
warnings.filterwarnings('ignore')
# 配置中文字体，请确保你的系统有这些字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class GardenAestheticsAnalyzer:
    """
    江南古典园林美学特征分析与建模器
    功能:
    1. 从平面图提取基础和空间结构特征。
    2. (可选)融合实景图的高级视觉特征。
    3. 计算园林间的相似度，并进行聚类分析。
    4. 将模型泛化应用于只有平面图的新园林。
    """
    def __init__(self, data_dir="赛题F江南古典园林美学特征建模附件资料"):
        self.data_dir = data_dir
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

        # 定义平面图的颜色分割阈值 (HSV色彩空间)
        # 这是一个关键部分，可能需要根据实际平面图微调
        self.color_ranges = {
            '水体': ([90, 80, 80], [130, 255, 255]),      # 蓝色范围
            '植物': ([35, 43, 46], [85, 255, 255]),      # 绿色范围
            '建筑': ([0, 0, 150], [180, 30, 255]),       # 灰色/深色范围
            '假山': ([10, 30, 100], [30, 150, 200]),     # 棕色/土黄色范围
            '道路': ([0, 0, 200], [180, 20, 255]),       # 白色/亮灰色范围
        }

        self.gardens_info = {
            '拙政园': {'id': 1}, '留园': {'id': 2}, '寄畅园': {'id': 3},
            '瞻园': {'id': 4}, '豫园': {'id': 5}, '秋霞圃': {'id': 6},
            '沈园': {'id': 7}, '怡园': {'id': 8}, '耦园': {'id': 9}, '绮园': {'id': 10}
        }

        self.feature_df = None

    def _find_file(self, directory, keywords):
        """
        在目录中查找包含所有关键词的文件。
        增强了对 '1-园林名称' 格式的匹配。
        """
        for fname in os.listdir(directory):
            # 同时检查文件名是否以 'id-' 开头，或者包含园林名称
            if all(kw in fname for kw in keywords):
                return os.path.join(directory, fname)
        return None

    def extract_features_from_plan(self, image_path):
        """
        从单张园林平面图中提取L1和L2特征
        L1: 基础面积占比特征
        L2: 空间结构特征
        【关键修改】: 使用 imdecode 来处理中文路径
        """
        if not os.path.exists(image_path):
            print(f"❌ 警告: 平面图文件不存在 {image_path}")
            return None

        # 使用可以处理中文路径的方式读取图片
        try:
            img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("图片文件为空或已损坏")
        except Exception as e:
            print(f"❌ 错误: 无法读取图片 {image_path}。原因: {e}")
            return None

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        height, width, _ = img_bgr.shape
        total_pixels = height * width

        features = {}
        masks = {}

        # L1 特征: 元素面积占比
        for element, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(img_hsv, np.array(lower), np.array(upper))

            # 基础形态学处理，去噪点，连接区域
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            masks[element] = mask
            features[f'{element}_占比'] = np.sum(mask > 0) / total_pixels

        # L2 特征: 空间结构特征
        # 1. 元素破碎度 (Fragmentation) - 以水体和假山为例
        for element in ['水体', '假山']:
            if features[f'{element}_占比'] > 0.001: # 仅在元素存在时计算
                num_labels, _ = measure.label(masks[element], connectivity=2, return_num=True)
                # 归一化破碎度：斑块数 / 元素面积 (乘以1000使数值更易读)
                features[f'{element}_破碎度'] = (num_labels -1) / (features[f'{element}_占比'] * total_pixels + 1e-6) * 1000
            else:
                features[f'{element}_破碎度'] = 0

        # 2. 空间复杂度: 边缘密度 (Edge Density)
        # 计算所有元素的总边缘长度
        total_edge_pixels = 0
        for mask in masks.values():
            edges = cv2.Canny(mask, 100, 200)
            total_edge_pixels += np.sum(edges > 0)
        features['边缘密度'] = total_edge_pixels / total_pixels

        # 3. 开合对比度: (水体+道路) vs (建筑+植物+假山)
        open_mask = cv2.bitwise_or(masks.get('水体', 0), masks.get('道路', 0))
        closed_mask = cv2.bitwise_or(masks.get('建筑', 0), masks.get('植物', 0))
        closed_mask = cv2.bitwise_or(closed_mask, masks.get('假山', 0))

        open_area = np.sum(open_mask > 0) / total_pixels
        closed_area = np.sum(closed_mask > 0) / total_pixels
        features['开合比'] = open_area / (closed_area + 1e-6)

        return features

    def process_all_gardens(self):
        """处理10个代表性园林，提取特征"""
        print("🚀 开始处理10个代表性园林...")
        all_features = []

        for name, info in tqdm(self.gardens_info.items(), desc="提取特征"):
            garden_id = info['id']
            # 路径示例: "赛题F.../1. 拙政园"
            garden_dir = os.path.join(self.data_dir, f"{garden_id}. {name}")

            if not os.path.exists(garden_dir):
                print(f"❓ 找不到目录: {garden_dir}")
                continue

            # 【关键修改】: 查找 '1-拙政园平面图.jpg' 或 '平面图.jpg'
            plan_image_path = self._find_file(garden_dir, [str(garden_id), name, '平面图', '.jpg'])
            if not plan_image_path:
                plan_image_path = self._find_file(garden_dir, ['平面图', '.jpg'])


            if plan_image_path:
                features = self.extract_features_from_plan(plan_image_path)
                if features:
                    features['园林名称'] = name
                    all_features.append(features)
            else:
                print(f"❌ 未找到 {name} 的平面图文件。请检查路径和文件名: {garden_dir}")

        self.feature_df = pd.DataFrame(all_features).set_index('园林名称')
        self.feature_df = self.feature_df.fillna(0) # 确保没有NaN值

        # 特征标准化
        self.feature_df_normalized = (self.feature_df - self.feature_df.mean()) / (self.feature_df.std() + 1e-6)

        # 保存特征
        self.feature_df.to_csv(os.path.join(self.output_dir, "features.csv"), encoding='utf-8-sig')
        print("✅ 10个园林的特征已提取并保存。")
        print(self.feature_df.head())

    def analyze_similarity(self):
        """分析园林相似度，并可视化"""
        if self.feature_df is None or self.feature_df.empty:
            print("特征提取失败，无法进行相似度分析。")
            return

        print("\n🔬 开始进行相似度分析...")
        # 1. 计算余弦相似度矩阵
        # scipy的pdist计算的是余弦距离 (1 - similarity)，所以需要转换
        cos_dist_matrix = squareform(pdist(self.feature_df_normalized, 'cosine'))
        similarity_matrix = 1 - cos_dist_matrix
        sim_df = pd.DataFrame(similarity_matrix, index=self.feature_df.index, columns=self.feature_df.index)

        # 2. 可视化相似度热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_df, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.title('江南古典园林美学特征相似度矩阵', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "similarity_matrix.png"), dpi=300)
        print("💾 相似度矩阵热力图已保存。")

        # 3. 层次聚类分析
        linked = linkage(self.feature_df_normalized, method='ward')

        plt.figure(figsize=(12, 8))
        dendrogram(linked,
                   orientation='top',
                   labels=self.feature_df.index.tolist(),
                   distance_sort='descending',
                   show_leaf_counts=True)
        plt.title('园林美学特征层次聚类', fontsize=16)
        plt.ylabel('距离')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "dendrogram.png"), dpi=300)
        print("💾 层次聚类树状图已保存。")

        # 4. 分析共性规律
        self._analyze_commonalities(sim_df)

    def _analyze_commonalities(self, sim_df):
        """从特征和相似度中解读共性规律"""
        print("\n📜 分析共有的美学特征与普遍性规律:")

        # 计算平均特征向量，即“标准江南园林”
        avg_garden = self.feature_df.mean()
        print("\n--- “标准”江南园林量化特征 ---")
        print(avg_garden)
        print("\n解读:")
        print(f"  - 平均来看，植物占比最高({avg_garden['植物_占比']:.2%})，其次是水体({avg_garden['水体_占比']:.2%})和建筑({avg_garden['建筑_占比']:.2%})。")
        print(f"  - 开合比({avg_garden['开合比']:.2f})大于1，说明整体上开放空间（水、路）略多于围合空间（建、植、山），追求疏朗感。")
        print(f"  - 水体和假山都有一定的破碎度({avg_garden['水体_破碎度']:.2f}, {avg_garden['假山_破碎度']:.2f})，体现了水随山转、山因水活的布局手法，而非大片单调的元素。")
        print(f"  - 较高的边缘密度({avg_garden['边缘密度']:.3f})表明元素间交错复杂，这正是“曲折尽致”和“移步异景”的量化体现。")

        # 找出最相似的园林对
        np.fill_diagonal(sim_df.values, 0)
        most_similar_pair = sim_df.stack().idxmax()
        similarity_score = sim_df.stack().max()
        print(f"\n--- 最相似的园林对 ---")
        print(f"  {most_similar_pair[0]} 与 {most_similar_pair[1]} 相似度最高，得分为: {similarity_score:.3f}")
        print("  这可能意味着它们在空间布局、元素配比和营造的意境上非常接近。")

    def generalize_to_new_garden(self, new_garden_plan_path):
        """
        将模型应用到新的园林，验证广效用
        """
        if self.feature_df is None or self.feature_df.empty:
            print("请先成功对10个代表性园林进行分析。")
            return

        garden_name = os.path.splitext(os.path.basename(new_garden_plan_path))[0]
        print(f"\n\n🚀 开始泛化应用到新园林: {garden_name}")

        # 1. 提取新园林的特征
        new_features = self.extract_features_from_plan(new_garden_plan_path)
        if not new_features:
            print(f"❌ 无法为 {garden_name} 提取特征。")
            return

        new_features_s = pd.Series(new_features, name=garden_name)

        # 2. 使用原数据集的均值和标准差进行标准化
        new_features_normalized = (new_features_s - self.feature_df.mean()) / (self.feature_df.std() + 1e-6)

        # 3. 计算与10个代表性园林的相似度
        similarities = {}
        for idx, row in self.feature_df_normalized.iterrows():
            # 计算余弦相似度
            sim = 1 - pdist([new_features_normalized.values, row.values], 'cosine')[0]
            similarities[idx] = sim

        sorted_sim = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

        # 4. 生成并展示验证报告
        report = f"--- “{garden_name}” 的广效用验证报告 ---\n\n"
        report += "1. 提取的美学特征:\n"
        report += new_features_s.to_string() + "\n\n"
        report += "2. 与十大代表园林的相似度排名:\n"
        for name, score in sorted_sim:
            report += f"   - 与 {name:<4s} 的相似度: {score:.3f}\n"

        most_similar_garden = sorted_sim[0][0]
        report += f"\n3. 结论:\n"
        report += f"   “{garden_name}” 在美学特征上与 “{most_similar_garden}” 最为接近。\n"
        report += f"   这表明它们的空间组织形式、元素构成比例和营造的“开合”感受可能非常相似。\n"
        report += f"   例如，可以比较它们的水体占比、边缘密度等具体指标来深入分析。\n"

        print(report)
        with open(os.path.join(self.output_dir, f"generalization_report_{garden_name}.txt"), 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 {garden_name} 的泛化报告已保存。")
