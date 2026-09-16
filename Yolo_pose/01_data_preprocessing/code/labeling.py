import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
try:
    import config
except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    import config
# ================================================
# ?ÑÎ°ú?ùÌä∏: ?âÏ? ?êÏÑ∏ Î∂ÑÎ•ò Î™®Îç∏ Í∞úÎ∞ú
# ?®Í≥Ñ: 1?®Í≥Ñ - ?∞Ïù¥???ÑÏ≤òÎ¶?
# ?§Î™Ö: CSV ?åÏùº Î≥ëÌï© Î∞?GOOD/BAD ?ºÎ≤®Îß??òÌñâ
# ?ëÏÑ±?? 2026.05.13
# ================================================
import pandas as pd
import os

# Paths
new_path = str(config.DATA_DIR / "new_images_data")
old_path = str(config.DATA_DIR / "FOR_DA")
output_path = str(config.ROOT_DIR)

def merge_csv(filename):
    new_file = os.path.join(new_path, filename)
    old_file = os.path.join(old_path, filename)
    
    if not os.path.exists(new_file) or not os.path.exists(old_file):
        print(f"Error: {filename} not found in one of the paths.")
        return None
    
    df_new = pd.read_csv(new_file)
    df_old = pd.read_csv(old_file)
    
    # concat - New data first to keep it in case of duplicates
    df_merged = pd.concat([df_new, df_old], ignore_index=True)
    
    # check duplicates in filename
    before_count = len(df_merged)
    # keep='first' maintains the row from df_new
    df_merged = df_merged.drop_duplicates(subset=['filename'], keep='first')
    after_count = len(df_merged)
    duplicates_removed = before_count - after_count
    
    # Save
    out_name = filename.replace('.csv', '_merged.csv')
    df_merged.to_csv(os.path.join(output_path, out_name), index=False)
    
    return {
        'n_new': len(df_new),
        'n_old': len(df_old),
        'n_dup': duplicates_removed,
        'n_total': len(df_merged),
        'df': df_merged
    }

# Merge final_labels_confirmed.csv
result_labels = merge_csv('final_labels_confirmed.csv')

# Merge yolo_landmarks_clean.csv
result_landmarks = merge_csv('yolo_landmarks_clean.csv')

if result_labels:
    n_new = result_labels['n_new']
    n_old = result_labels['n_old']
    n_dup = result_labels['n_dup']
    n_total = result_labels['n_total']
    df_final = result_labels['df']
    
    # Stats for final_labels
    if 'final_label' in df_final.columns:
        good_count = len(df_final[df_final['final_label'] == 'GOOD'])
        bad_count = len(df_final[df_final['final_label'] == 'BAD'])
        good_pct = (good_count / n_total) * 100 if n_total > 0 else 0
        bad_pct = (bad_count / n_total) * 100 if n_total > 0 else 0
        
        print(f"?†Í∑ú ?∞Ïù¥?? {n_new}??)
        print(f"Í∏∞Ï°¥ ?∞Ïù¥?? {n_old}??)
        print(f"Ï§ëÎ≥µ ?úÍ±∞: {n_dup}??)
        print(f"ÏµúÏ¢Ö ?©Í≥Ñ: {n_total}??)
        print(f"GOOD: {good_count}??({good_pct:.1f}%)")
        print(f"BAD: {bad_count}??({bad_pct:.1f}%)")
    else:
        print("Error: 'final_label' column not found in merged data.")


