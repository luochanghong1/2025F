from garden_analyzer import GardenAestheticsAnalyzer
import os

def find_attachment11_dir(base_data_dir):
    """
    Intelligently finds the directory for Attachment 11.
    Handles various possible naming conventions and cases where the folder might be missing.
    """
    print("\n--- Intelligently searching for Attachment 11 directory ---")

    if not os.path.exists(base_data_dir):
        print(f"❌ CRITICAL ERROR: The base data directory '{base_data_dir}' does not exist!")
        print(f"Please ensure the folder named '{base_data_dir}' is in the same location as your Python script.")
        print(f"Current working directory is: {os.getcwd()}")
        return None

    print(f"✅ Base data directory '{base_data_dir}' found.")
    print("  Listing its contents for debugging:")

    found_items = os.listdir(base_data_dir)
    for item in found_items:
        print(f"    - {item}")

    possible_names = [
        "11. 其他园林平面图",
        "11.其他园林平面图",
        "附件11",
    ]

    for name in possible_names:
        path = os.path.join(base_data_dir, name)
        if os.path.exists(path) and os.path.isdir(path):
            print(f"✅ Successfully matched Attachment 11 directory: '{path}'")
            return path

    print("\nStandard names not found, attempting a fuzzy search for directories starting with '11'...")
    for item in found_items:
        path = os.path.join(base_data_dir, item)
        if os.path.isdir(path) and item.strip().startswith('11'):
            print(f"✅ Successfully fuzzy-matched Attachment 11 directory: '{path}'")
            return path

    print("❌ ERROR: Could not find the Attachment 11 directory. Please check the folder name.")
    return None


def main():
    """
    Main execution function:
    1. Builds a multimodal model based on 10 representative gardens (plans + photos).
    2. Generalizes the model to all other gardens in Attachment 11 (plans only).
    """
    base_data_dir = "赛题F江南古典园林美学特征建模附件资料"
    analyzer = GardenAestheticsAnalyzer(data_dir=base_data_dir)

    # =========================================================================
    # Part 1: Multimodal Similarity Analysis (on 10 representative gardens)
    # =========================================================================
    print("="*80)
    print("Part 1: Multimodal Similarity Analysis and Pattern Discovery")
    print("="*80)

    analyzer.process_all_gardens()

    if analyzer.feature_df is not None and not analyzer.feature_df.empty:
        analyzer.analyze_similarity()
    else:
        print("❌ Feature extraction failed, cannot proceed with Part 1. Please check file paths and formats.")
        return

    # =========================================================================
    # Part 2: Generalization Validation (on all gardens in Attachment 11)
    # =========================================================================
    print("\n" + "="*80)
    print("Part 2: Model Generalization Validation")
    print("="*80)

    new_gardens_dir = find_attachment11_dir(base_data_dir)

    if new_gardens_dir is None:
        print("Cannot proceed with generalization validation as the Attachment 11 directory was not found.")
        return

    try:
        test_gardens_files = [f for f in os.listdir(new_gardens_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot access directory: {new_gardens_dir}")
        return

    if not test_gardens_files:
        print(f"⚠️ WARNING: No plan drawings (.jpg, .png) found in '{new_gardens_dir}'.")
        return

    print(f"\nFound {len(test_gardens_files)} plan drawings in the Attachment 11 directory. Analyzing each one...")
    print(f"File list: {test_gardens_files}")

    successful_tests = 0
    for garden_file in test_gardens_files:
        new_garden_path = os.path.join(new_gardens_dir, garden_file)
        analyzer.generalize_to_new_garden(new_garden_path)
        successful_tests += 1

    print(f"\n✅ Generalization validation complete. Successfully analyzed {successful_tests}/{len(test_gardens_files)} new gardens.")
    print("\n\n🎉 All analysis tasks are complete! Results have been saved in the 'results' folder.")

if __name__ == "__main__":
    main()
