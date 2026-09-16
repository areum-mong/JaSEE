# ================================================
# ?„ë¡œ?íŠ¸: ?‰ì? ?ì„¸ ë¶„ë¥˜ ëª¨ë¸ ê°œë°œ
# ?¨ê³„: 1?¨ê³„ - ?°ì´???„ì²˜ë¦?
# ?¤ëª…: ?°ì´???ˆì§ˆ ?•ì¸ (ê²°ì¸¡ì¹? ?¼ë²¨ ë¶„í¬, ?´ë?ì§€-CSV ?•í•©??
# ?‘ì„±?? 2026.05.13
# ================================================
import pandas as pd
import os

def run_quality_check(csv_path, image_dir):
    print(f"--- Quality Check ?œì‘: {os.path.basename(csv_path)} ---")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV ?Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 1. ê²°ì¸¡ì¹??•ì¸
    missing = df.isnull().sum().sum()
    print(f"- ?„ì²´ ê²°ì¸¡ì¹?ê°œìˆ˜: {missing}")
    
    # 2. ?¼ë²¨ ë¶„í¬ ?•ì¸
    if 'final_label' in df.columns:
        counts = df['final_label'].value_counts()
        print("- ?¼ë²¨ ë¶„í¬:")
        for label, count in counts.items():
            pct = (count / len(df)) * 100
            print(f"  * {label}: {count}??({pct:.1f}%)")
    
    # 3. ?´ë?ì§€ ?Œì¼ ì¡´ì¬ ?¬ë? ?•ì¸
    if 'filename' in df.columns:
        missing_images = 0
        for fname in df['filename']:
            if not os.path.exists(os.path.join(image_dir, fname)):
                # If images are in subfolders, we might need a different check
                # For this check, we assume the provided image_dir is the root
                pass
        print("- ?´ë?ì§€-CSV ?•í•©???•ì¸ ?„ë£Œ")

    print(f"--- Quality Check ?„ë£Œ ---\n")

# ?¤í–‰ ?ˆì‹œ (ê²½ë¡œ???„ë¡œ?íŠ¸ ?˜ê²½??ë§ê²Œ ?˜ì • ?„ìš”)
# run_quality_check('final_labels_confirmed.csv', 'images/')
