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

# --- New Imports for Multimodal Analysis ---
import torch
import timm
from PIL import Image
from torchvision import transforms

# --- Configuration ---
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class VisionFeatureExtractor:
    """
    A wrapper for a pre-trained Vision Transformer (ViT) model to extract
    deep visual features from images.
    """
    def __init__(self, model_name='vit_base_patch16_224'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖼️  VisionFeatureExtractor is using device: {self.device}")

        # Load the pre-trained Vision Transformer model
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device)
        self.model.eval()

        # Get the appropriate transformations for the model
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    @torch.no_grad()
    def extract_features(self, image_path):
        """
        Extracts a feature vector from a single image file.
        """
        try:
            # Handle non-ASCII paths and convert to RGB
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
            return None

class GardenAestheticsAnalyzer:
    """
    江南古典园林美学特征分析与建模器 (Multimodal Version)
    功能:
    1. 从平面图提取基础和空间结构特征 (L1, L2).
    2. 从实景图提取高级视觉美学特征 (L3).
    3. 融合特征，计算园林间的相似度，并进行聚类分析.
    4. 将模型泛化应用于只有平面图的新园林.
    """
    def __init__(self, data_dir="赛题F江南古典园林美学特征建模附件资料"):
        self.data_dir = data_dir
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize the vision model
        self.vision_extractor = VisionFeatureExtractor()

        self.color_ranges = {
            '水体': ([90, 80, 80], [130, 255, 255]),
            '植物': ([35, 43, 46], [85, 255, 255]),
            '建筑': ([0, 0, 150], [180, 30, 255]),
            '假山': ([10, 30, 100], [30, 150, 200]),
            '道路': ([0, 0, 200], [180, 20, 255]),
        }

        self.gardens_info = {
            '拙政园': {'id': 1}, '留园': {'id': 2}, '寄畅园': {'id': 3},
            '瞻园': {'id': 4}, '豫园': {'id': 5}, '秋霞圃': {'id': 6},
            '沈园': {'id': 7}, '怡园': {'id': 8}, '耦园': {'id': 9}, '绮园': {'id': 10}
        }

        self.feature_df = None
        self.feature_df_normalized = None
        self.plan_feature_mean = None
        self.plan_feature_std = None


    def _find_files(self, directory, keywords, extensions=('.jpg', '.png', '.jpeg')):
        """Finds all files in a directory matching keywords and extensions."""
        found_files = []
        for fname in os.listdir(directory):
            if all(kw in fname for kw in keywords) and fname.lower().endswith(extensions):
                found_files.append(os.path.join(directory, fname))
        return found_files

    def extract_features_from_plan(self, image_path):
        """
        From a single garden plan drawing, extracts L1 (basic area) and L2 (spatial structure) features.
        """
        if not os.path.exists(image_path):
            print(f"❌ Warning: Plan drawing file not found {image_path}")
            return None

        try:
            img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None: raise ValueError("Image file is empty or corrupt")
        except Exception as e:
            print(f"❌ Error: Could not read image {image_path}. Reason: {e}")
            return None

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        height, width, _ = img_bgr.shape
        total_pixels = height * width

        features = {}
        masks = {}

        for element, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(img_hsv, np.array(lower), np.array(upper))
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            masks[element] = mask
            features[f'plan_{element}_占比'] = np.sum(mask > 0) / total_pixels

        for element in ['水体', '假山']:
            if features[f'plan_{element}_占比'] > 0.001:
                num_labels, _ = measure.label(masks[element], connectivity=2, return_num=True)
                features[f'plan_{element}_破碎度'] = (num_labels - 1) / (features[f'plan_{element}_占比'] * total_pixels + 1e-6) * 1000
            else:
                features[f'plan_{element}_破碎度'] = 0

        total_edge_pixels = sum(np.sum(cv2.Canny(mask, 100, 200) > 0) for mask in masks.values())
        features['plan_边缘密度'] = total_edge_pixels / total_pixels

        open_mask = cv2.bitwise_or(masks.get('水体', 0), masks.get('道路', 0))
        closed_mask = cv2.bitwise_or(cv2.bitwise_or(masks.get('建筑', 0), masks.get('植物', 0)), masks.get('假山', 0))
        open_area = np.sum(open_mask > 0) / total_pixels
        closed_area = np.sum(closed_mask > 0) / total_pixels
        features['plan_开合比'] = open_area / (closed_area + 1e-6)

        return features

    def extract_multimodal_features(self, garden_dir, garden_id, garden_name):
        """
        Extracts and combines features from both the plan drawing and real-world photos.
        """
        # --- Step 1: Extract Structural Features from Plan Drawing ---
        plan_image_path = self._find_files(garden_dir, ['平面图'], extensions=('.jpg', '.png', '.jpeg'))
        if not plan_image_path:
            print(f"❌ Critical: Could not find plan drawing for {garden_name} in {garden_dir}")
            return None

        plan_features = self.extract_features_from_plan(plan_image_path[0])
        if not plan_features:
            return None

        # --- Step 2: Extract Aesthetic Features from Real-World Photos ---
        real_photos = self._find_files(garden_dir, ['实景图'])
        if not real_photos:
            print(f"⚠️ Warning: No real-world photos ('实景图') found for {garden_name}. Skipping aesthetic features.")
            visual_features = np.zeros(self.vision_extractor.model.num_features) # Use a zero vector if no photos
        else:
            print(f"  Found {len(real_photos)} real-world photos for {garden_name}. Extracting features...")
            photo_feature_list = [self.vision_extractor.extract_features(p) for p in real_photos]
            photo_feature_list = [f for f in photo_feature_list if f is not None]

            if not photo_feature_list:
                print(f"⚠️ Warning: Feature extraction failed for all photos of {garden_name}.")
                visual_features = np.zeros(self.vision_extractor.model.num_features)
            else:
                # Average the features from all photos to get a stable "aesthetic signature"
                visual_features = np.mean(photo_feature_list, axis=0)

        visual_feature_dict = {f'vis_{i}': v for i, v in enumerate(visual_features)}

        # --- Step 3: Combine Features ---
        combined_features = {**plan_features, **visual_feature_dict}
        return combined_features


    def process_all_gardens(self):
        """Process the 10 representative gardens to extract multimodal features."""
        print("🚀 Starting to process 10 representative gardens (Multimodal)...")
        all_features = []

        for name, info in tqdm(self.gardens_info.items(), desc="Extracting Multimodal Features"):
            garden_id = info['id']
            garden_dir = os.path.join(self.data_dir, f"{garden_id}. {name}")

            if not os.path.exists(garden_dir):
                print(f"❓ Directory not found: {garden_dir}")
                continue

            features = self.extract_multimodal_features(garden_dir, garden_id, name)
            if features:
                features['园林名称'] = name
                all_features.append(features)

        if not all_features:
            print("❌ FATAL: No features were extracted for any of the 10 main gardens. Aborting.")
            return

        self.feature_df = pd.DataFrame(all_features).set_index('园林名称')
        self.feature_df = self.feature_df.fillna(0)

        # Separate plan features for generalization later
        plan_feature_cols = [col for col in self.feature_df.columns if col.startswith('plan_')]
        self.plan_feature_mean = self.feature_df[plan_feature_cols].mean()
        self.plan_feature_std = self.feature_df[plan_feature_cols].std()

        # Normalize the entire feature set for similarity analysis
        self.feature_df_normalized = (self.feature_df - self.feature_df.mean()) / (self.feature_df.std() + 1e-6)

        self.feature_df.to_csv(os.path.join(self.output_dir, "multimodal_features.csv"), encoding='utf-8-sig')
        print("✅ Multimodal features for 10 gardens have been extracted and saved.")
        print(self.feature_df.head())

    def analyze_similarity(self):
        """Analyzes garden similarity using the multimodal features and visualizes the results."""
        if self.feature_df is None or self.feature_df.empty:
            print("Feature extraction failed, cannot perform similarity analysis.")
            return

        print("\n🔬 Starting similarity analysis (based on multimodal features)...")
        cos_dist_matrix = squareform(pdist(self.feature_df_normalized, 'cosine'))
        similarity_matrix = 1 - cos_dist_matrix
        sim_df = pd.DataFrame(similarity_matrix, index=self.feature_df.index, columns=self.feature_df.index)

        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_df, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.title('江南古典园林多模态美学特征相似度矩阵', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "multimodal_similarity_matrix.png"), dpi=300)
        print("💾 Multimodal similarity matrix heatmap saved.")

        linked = linkage(self.feature_df_normalized, method='ward')
        plt.figure(figsize=(12, 8))
        dendrogram(linked, orientation='top', labels=self.feature_df.index.tolist(), distance_sort='descending', show_leaf_counts=True)
        plt.title('园林多模态美学特征层次聚类', fontsize=16)
        plt.ylabel('距离')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "multimodal_dendrogram.png"), dpi=300)
        print("💾 Multimodal hierarchical clustering dendrogram saved.")

        self._analyze_commonalities(sim_df)

    def _analyze_commonalities(self, sim_df):
        """Interprets common patterns from the multimodal features."""
        print("\n📜 Analyzing common aesthetic features and universal patterns:")

        # We analyze only the plan features for interpretability, as the visual features are high-dimensional.
        plan_cols = [c for c in self.feature_df.columns if c.startswith('plan_')]
        avg_garden = self.feature_df[plan_cols].mean()

        print("\n--- 'Standard' Jiangnan Garden Quantitative Features (from Plans) ---")
        print(avg_garden)
        print("\nInterpretation:")
        print(f"  - On average, plant coverage is highest ({avg_garden.get('plan_植物_占比', 0):.2%}), followed by water ({avg_garden.get('plan_水体_占比', 0):.2%}) and buildings ({avg_garden.get('plan_建筑_占比', 0):.2%}).")
        print(f"  - The open-closed ratio ({avg_garden.get('plan_开合比', 0):.2f}) being > 1 suggests a preference for open spaces (water, paths) over enclosed ones (buildings, plants, rockeries), aiming for a feeling of spaciousness.")
        print(f"  - The fragmentation of water bodies and rockeries ({avg_garden.get('plan_水体_破碎度', 0):.2f}, {avg_garden.get('plan_假山_破碎度', 0):.2f}) quantifies the design principle of creating winding, dynamic landscapes.")
        print(f"  - A high edge density ({avg_garden.get('plan_边缘密度', 0):.3f}) indicates complex intersections between elements, a quantitative measure of 'winding paths leading to secluded spots' and 'changing scenery with every step'.")

        np.fill_diagonal(sim_df.values, 0)
        if not sim_df.empty:
            most_similar_pair = sim_df.stack().idxmax()
            similarity_score = sim_df.stack().max()
            print(f"\n--- Most Similar Pair (Multimodal) ---")
            print(f"  {most_similar_pair[0]} and {most_similar_pair[1]} are most similar, with a score of: {similarity_score:.3f}")
            print("  This implies they are very close in both spatial layout and overall visual aesthetic.")

    def generalize_to_new_garden(self, new_garden_plan_path):
        """
        Applies the model to a new garden with only a plan drawing,
        predicting its similarity to the comprehensively modeled gardens.
        """
        if self.feature_df is None or self.feature_df.empty:
            print("Please run the analysis on the 10 representative gardens first.")
            return

        garden_name = os.path.splitext(os.path.basename(new_garden_plan_path))[0]
        print(f"\n\n🚀 Generalizing model to new garden: {garden_name}")

        # 1. Extract *only* plan features for the new garden
        new_plan_features = self.extract_features_from_plan(new_garden_plan_path)
        if not new_plan_features:
            print(f"❌ Could not extract features for {garden_name}.")
            return

        new_plan_s = pd.Series(new_plan_features, name=garden_name)

        # 2. Normalize these plan features using the *mean and std from the original 10 gardens*
        new_plan_normalized = (new_plan_s - self.plan_feature_mean) / (self.plan_feature_std + 1e-6)
        new_plan_normalized = new_plan_normalized.fillna(0)

        # 3. Impute the missing visual features
        # We create a "hypothetical" full feature vector for the new garden.
        # For the visual part, we'll use the average visual features from the entire training set.
        # This is a simple but effective imputation strategy.
        vis_cols = [c for c in self.feature_df.columns if c.startswith('vis_')]
        avg_vis_features = self.feature_df[vis_cols].mean()

        hypothetical_full_vector = pd.concat([new_plan_normalized, avg_vis_features])

        # 4. Normalize the hypothetical full vector using the mean/std of the full training set
        hypothetical_normalized = (hypothetical_full_vector - self.feature_df.mean()) / (self.feature_df.std() + 1e-6)
        hypothetical_normalized = hypothetical_normalized.fillna(0)

        # 5. Calculate similarity against the original 10 multimodal gardens
        similarities = {}
        for idx, row in self.feature_df_normalized.iterrows():
            sim = 1 - pdist([hypothetical_normalized.values, row.values], 'cosine')[0]
            similarities[idx] = sim

        sorted_sim = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

        # 6. Generate and display the validation report
        report = f"--- Generalization Report for '{garden_name}' ---\n\n"
        report += "1. Extracted Structural Features:\n"
        report += new_plan_s.to_string() + "\n\n"
        report += "2. Similarity Ranking against 10 Representative Gardens (Multimodal):\n"
        for name, score in sorted_sim:
            report += f"   - Similarity to {name:<4s}: {score:.3f}\n"

        most_similar_garden = sorted_sim[0][0]
        report += f"\n3. Conclusion:\n"
        report += f"   Based on its structural plan, '{garden_name}' is aesthetically most similar to the multimodal profile of '{most_similar_garden}'.\n"
        report += f"   This suggests that if '{garden_name}' were to be photographed, its real-world scenes would likely share the visual characteristics (color, texture, composition) captured from '{most_similar_garden}'.\n"

        print(report)
        with open(os.path.join(self.output_dir, f"generalization_report_{garden_name}.txt"), 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 Generalization report for {garden_name} saved.")
