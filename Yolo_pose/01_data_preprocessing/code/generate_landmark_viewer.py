import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
try:
    import config
except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    import config
import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ?¼ì´ë¸ŒëŸ¬ë¦?ì²´í¬ ë°??¤ì¹˜
try:
    import cv2
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("?„ìš”???¼ì´ë¸ŒëŸ¬ë¦¬ë? ?¤ì¹˜?©ë‹ˆ??..")
    install("opencv-python")
    install("pandas")
    install("numpy")
    install("tqdm")
    import cv2
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

import base64

def generate_viewer(csv_path, image_dirs, output_html):
    print(f"--- Landmark Viewer ?ì„± ?œì‘ ---")
    
    if not os.path.exists(csv_path):
        print(f"[?¤ë¥˜] CSV ?Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # ?‰ìƒ ?•ì˜ (BGR)
    COLORS = {
        'nose': (255, 255, 255),    # ?°ìƒ‰
        'eye': (235, 206, 135),     # ?˜ëŠ˜??
        'ear': (0, 255, 255),       # ?¸ë???
        'shoulder': (255, 0, 0),    # ?Œë???
        'elbow': (0, 165, 255),     # ì£¼í™©??
        'wrist': (203, 192, 255),   # ë¶„í™??
        'hip': (0, 255, 0),         # ì´ˆë¡??
        'knee': (128, 0, 128),      # ë³´ë¼??
        'ankle': (0, 0, 255)        # ë¹¨ê°„??
    }

    CONNECTIONS = [
        (('nose_x', 'nose_y'), ('left_shoulder_x', 'left_shoulder_y'), COLORS['nose']),
        (('nose_x', 'nose_y'), ('right_shoulder_x', 'right_shoulder_y'), COLORS['nose']),
        (('left_ear_x', 'left_ear_y'), ('left_shoulder_x', 'left_shoulder_y'), COLORS['ear']),
        (('right_ear_x', 'right_ear_y'), ('right_shoulder_x', 'right_shoulder_y'), COLORS['ear']),
        (('left_shoulder_x', 'left_shoulder_y'), ('left_hip_x', 'left_hip_y'), COLORS['shoulder']),
        (('right_shoulder_x', 'right_shoulder_y'), ('right_hip_x', 'right_hip_y'), COLORS['shoulder']),
        (('left_hip_x', 'left_hip_y'), ('left_knee_x', 'left_knee_y'), COLORS['hip']),
        (('right_hip_x', 'right_hip_y'), ('right_knee_x', 'right_knee_y'), COLORS['hip']),
        (('left_knee_x', 'left_knee_y'), ('left_ankle_x', 'left_ankle_y'), COLORS['ankle']),
        (('right_knee_x', 'right_knee_y'), ('right_ankle_x', 'right_ankle_y'), COLORS['ankle']),
        (('left_shoulder_x', 'left_shoulder_y'), ('left_elbow_x', 'left_elbow_y'), COLORS['elbow']),
        (('right_shoulder_x', 'right_shoulder_y'), ('right_elbow_x', 'right_elbow_y'), COLORS['elbow']),
        (('left_elbow_x', 'left_elbow_y'), ('left_wrist_x', 'left_wrist_y'), COLORS['wrist']),
        (('right_elbow_x', 'right_elbow_y'), ('right_wrist_x', 'right_wrist_y'), COLORS['wrist'])
    ]

    KP_MAP = {
        'nose': 'nose', 'left_eye': 'eye', 'right_eye': 'eye',
        'left_ear': 'ear', 'right_ear': 'ear', 'left_shoulder': 'shoulder',
        'right_shoulder': 'shoulder', 'left_elbow': 'elbow', 'right_elbow': 'elbow',
        'left_wrist': 'wrist', 'right_wrist': 'wrist', 'left_hip': 'hip',
        'right_hip': 'hip', 'left_knee': 'knee', 'right_knee': 'knee',
        'left_ankle': 'ankle', 'right_ankle': 'ankle'
    }

    html_cards = []
    stats = {'full': 0, 'partial': 0, 'flipped': 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="?´ë?ì§€ ì²˜ë¦¬ ì¤?):
        filename = row['filename']
        img_path = None
        for d in image_dirs:
            # Recursively check subfolders if needed, but here we assume direct or depth-1
            trial = os.path.join(d, filename)
            if os.path.exists(trial):
                img_path = trial
                break
            # Try GOOD/BAD subfolders
            for sub in ['GOOD', 'BAD']:
                trial = os.path.join(d, sub, filename)
                if os.path.exists(trial):
                    img_path = trial
                    break
        
        if not img_path:
            continue

        image = cv2.imread(img_path)
        if image is None: continue
        
        if row.get('is_flipped', False):
            image = cv2.flip(image, 1)
            stats['flipped'] += 1

        h, w = image.shape[:2]
        recognized_count = 0

        # ??ê·¸ë¦¬ê¸?
        for start_kp, end_kp, color in CONNECTIONS:
            x1, y1 = row[start_kp[0]] * w, row[start_kp[1]] * h
            x2, y2 = row[end_kp[0]] * w, row[end_kp[1]] * h
            c1 = row[start_kp[0].replace('_x', '_conf')]
            c2 = row[end_kp[0].replace('_x', '_conf')]
            
            if c1 >= 0.5 and c2 >= 0.5:
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # ??ê·¸ë¦¬ê¸?
        for kp, type_name in KP_MAP.items():
            x, y = row[f'{kp}_x'] * w, row[f'{kp}_y'] * h
            conf = row[f'{kp}_conf']
            color = COLORS[type_name]
            
            if conf >= 0.5:
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
                recognized_count += 1
            elif conf > 0:
                # ?ì„  ?€???‡ì? ?ìœ¼ë¡??œì‹œ
                cv2.circle(image, (int(x), int(y)), 5, color, 1)

        if recognized_count == 17: stats['full'] += 1
        else: stats['partial'] += 1

        # ?ìŠ¤???¤ë²„?ˆì´
        cv2.putText(image, f"sitting_direction: {row['sitting_direction']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"is_flipped: {row['is_flipped']}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Base64 ë³€??
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # ì¹´ë“œ ?ì„±
        card = f"""
        <div class="card" data-dir="{row['sitting_direction']}" data-flipped="{row['is_flipped']}" data-full="{recognized_count == 17}">
            <img src="data:image/jpeg;base64,{img_base64}">
            <div class="info">
                <div class="filename">{filename}</div>
                <div>ë°©í–¥: {row['sitting_direction']} | ë°˜ì „: {row['is_flipped']}</div>
                <div class="recon">?¸ì‹ ê´€?? {recognized_count}/17ê°?/div>
            </div>
        </div>
        """
        html_cards.append(card)

    # HTML ?œí”Œë¦?
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>Landmark Viewer</title>
        <style>
            body {{ background-color: #0f172a; color: #f8fafc; font-family: sans-serif; margin: 2rem; }}
            .stats {{ background: #1e293b; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; display: flex; gap: 2rem; }}
            .filters {{ margin-bottom: 2rem; display: flex; gap: 1rem; }}
            button {{ background: #334155; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer; }}
            button.active {{ background: #6366f1; }}
            .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; }}
            .card {{ background: #1e293b; border-radius: 0.5rem; overflow: hidden; border: 1px solid #334155; }}
            .card img {{ width: 100%; display: block; }}
            .card .info {{ padding: 1rem; }}
            .filename {{ font-weight: bold; margin-bottom: 0.5rem; color: #22d3ee; word-break: break-all; }}
            .recon {{ color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem; }}
        </style>
    </head>
    <body>
        <h1>Pose Landmark Viewer</h1>
        
        <div class="stats">
            <div>?„ì²´ ?´ë?ì§€: <b>{len(df)}??/b></div>
            <div>17ê°??„ì „ ?¸ì‹: <b style="color:#22d3ee">{stats['full']}??/b></div>
            <div>ë¶€ë¶??¸ì‹: <b style="color:#f87171">{stats['partial']}??/b></div>
            <div>ë°˜ì „ ì²˜ë¦¬: <b style="color:#fbbf24">{stats['flipped']}??/b></div>
        </div>

        <div class="filters">
            <button onclick="filter('all')" class="active">?„ì²´ ë³´ê¸°</button>
            <button onclick="filter('LEFT')">LEFTë§?/button>
            <button onclick="filter('RIGHT')">RIGHTë§?/button>
            <button onclick="filter('flipped')">ë°˜ì „ ?´ë?ì§€ë§?/button>
            <button onclick="filter('partial')">ë¶€ë¶„ì¸?ë§Œ</button>
        </div>

        <div class="grid" id="grid">
            {''.join(html_cards)}
        </div>

        <script>
            function filter(type) {{
                const cards = document.querySelectorAll('.card');
                const buttons = document.querySelectorAll('button');
                buttons.forEach(b => b.classList.remove('active'));
                event.target.classList.add('active');

                cards.forEach(card => {{
                    card.style.display = 'none';
                    if (type === 'all') card.style.display = 'block';
                    else if (type === 'LEFT' && card.dataset.dir === 'LEFT') card.style.display = 'block';
                    else if (type === 'RIGHT' && card.dataset.dir === 'RIGHT') card.style.display = 'block';
                    else if (type === 'flipped' && card.dataset.flipped === 'True') card.style.display = 'block';
                    else if (type === 'partial' && card.dataset.full === 'false') card.style.display = 'block';
                }});
            }}
        </script>
    </body>
    </html>
    """

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”")
    print(f" ?„ì²´ ?´ë?ì§€: {len(df)}??)
    print(f" ?œê°???„ë£Œ: {len(html_cards)}??)
    print(f" ?€??ê²½ë¡œ: {output_html}")
    print(f"?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”?â”")

if __name__ == "__main__":
    # ?„ë¡œ?íŠ¸ ?˜ê²½??ë§ëŠ” ê²½ë¡œ ?¤ì •
    CSV_FILE = r'yolo_landmarks_extracted.csv'
    IMG_DIRS = [
        str(config.DATA_DIR / "new_images_data/images"),
        str(config.DATA_DIR / "FOR_DA/YOLO_full_body"),
        str(config.DATA_DIR / "FOR_DA/YOLO_ankle_visible")
    ]
    OUTPUT = 'landmark_viewer.html'
    
    generate_viewer(CSV_FILE, IMG_DIRS, OUTPUT)


