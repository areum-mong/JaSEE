import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
try:
    import config
except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    import config
# ================================================
# ?„ë¡œ?íŠ¸: ?‰ì? ?ì„¸ ë¶„ë¥˜ ëª¨ë¸ ê°œë°œ
# ?¨ê³„: 1?¨ê³„ - ?°ì´???„ì²˜ë¦?
# ?¤ëª…: YOLOv8-Pose ê³ ë„???œë“œë§ˆí¬ ì¶”ì¶œ (ì¢Œìš° ë°˜ì „ ë°?ë°©í–¥ ?ì • ?¬í•¨)
# ?‘ì„±?? 2026.05.14
# ================================================
from ultralytics import YOLO
import pandas as pd
import os
import glob
import cv2
import numpy as np

def extract_landmarks(image_dir, model_path='yolov8n-pose.pt'):
    print(f"--- Landmark ì¶”ì¶œ ?œì‘: {image_dir} ---")
    
    model = YOLO(model_path)
    images = glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True) + \
             glob.glob(os.path.join(image_dir, "**/*.png"), recursive=True)
    
    results_list = []
    failed_images = []
    flip_count = 0
    
    kp_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    for img_path in images:
        fname = os.path.basename(img_path)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            failed_images.append(f"{fname} (?Œì¼ ë¡œë“œ ?¤íŒ¨)")
            continue

        # 1ì°?ì¶”ë¡  (ë°©í–¥ ë°?ë°˜ì „ ?¬ë? ?ë‹¨??
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = model(image_rgb, verbose=False)
        
        target_res = None
        for r in results:
            if r.keypoints is not None and len(r.keypoints.conf) > 0:
                target_res = r
                break
        
        if target_res is None:
            failed_images.append(f"{fname} (?¬ëŒ ê°ì? ?¤íŒ¨)")
            continue

        conf = target_res.keypoints.conf[0].cpu().numpy()
        left_ear_conf = conf[3]
        right_ear_conf = conf[4]
        
        is_flipped = False
        # ?¼ìª½ ì¸¡ë©´ ?´ë?ì§€ ?ë‹¨ (?¼ìª½ ê·€ê°€ ????ë³´ì¼ ??
        if right_ear_conf < left_ear_conf:
            image_bgr = cv2.flip(image_bgr, 1) # ì¢Œìš° ë°˜ì „
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = model(image_rgb, verbose=False)
            
            target_res = None
            for r in results:
                if r.keypoints is not None and len(r.keypoints.conf) > 0:
                    target_res = r
                    break
            
            if target_res is None:
                failed_images.append(f"{fname} (ë°˜ì „ ??ê°ì? ?¤íŒ¨)")
                continue
            
            conf = target_res.keypoints.conf[0].cpu().numpy()
            is_flipped = True
            flip_count += 1

        # ê´€??ì¢Œí‘œ ë°?? ë¢°??ì¶”ì¶œ
        kpts = target_res.keypoints.xyn[0].cpu().numpy()
        
        l_sh_conf = conf[5]
        r_sh_conf = conf[6]
        
        # ê°ì? ?¤íŒ¨ ì²˜ë¦¬ (?´ê¹¨ ? ë¢°??ê¸°ì?)
        if l_sh_conf < 0.5 or r_sh_conf < 0.5:
            failed_images.append(f"{fname} (?´ê¹¨ ? ë¢°????Œ: L={l_sh_conf:.2f}, R={r_sh_conf:.2f})")
            continue

        # ?‰ì? ë°©í–¥ ?ì •
        nose_x = kpts[0][0]
        l_sh_x = kpts[5][0]
        r_sh_x = kpts[6][0]
        shoulder_center_x = (l_sh_x + r_sh_x) / 2
        
        sitting_direction = "LEFT" if nose_x < shoulder_center_x else "RIGHT"

        # ê²°ê³¼ ?€??
        row = {
            'filename': fname,
            'sitting_direction': sitting_direction,
            'is_flipped': is_flipped
        }
        
        for i, name in enumerate(kp_names):
            row[f'{name}_x'] = kpts[i][0]
            row[f'{name}_y'] = kpts[i][1]
            row[f'{name}_conf'] = conf[i]
            
        results_list.append(row)

    # CSV ?€??
    df = pd.DataFrame(results_list)
    output_csv = 'yolo_landmarks_extracted.csv'
    df.to_csv(output_csv, index=False)
    
    # ?¤íŒ¨ ëª©ë¡ ?€??
    with open('failed_images.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(failed_images))

    # ì½˜ì†” ì¶œë ¥
    print(f"?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”")
    print(f" ?„ì²´ ?´ë?ì§€: {len(images)}??)
    print(f" ì¶”ì¶œ ?±ê³µ: {len(results_list)}??)
    print(f" ê°ì? ?¤íŒ¨: {len(failed_images)}??)
    if failed_images:
        print(f" [?¤íŒ¨ ëª©ë¡]")
        for f in failed_images:
            print(f"  - {f}")
    print(f" ?¼ìª½ ì¸¡ë©´ ë°˜ì „ ì²˜ë¦¬: {flip_count}??)
    print(f"?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”")
    print(f" ê²°ê³¼ê°€ '{output_csv}' ë°?'failed_images.txt'???€?¥ë˜?ˆìŠµ?ˆë‹¤.")

# ?¤í–‰ ?ˆì‹œ (?„ìš” ??ì£¼ì„ ?´ì œ)
# extract_landmarks(str(config.DATA_DIR / "images"))

