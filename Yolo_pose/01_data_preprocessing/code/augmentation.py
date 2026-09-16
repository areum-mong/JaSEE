# ================================================
# ?„ë¡œ?íŠ¸: ?‰ì? ?ì„¸ ë¶„ë¥˜ ëª¨ë¸ ê°œë°œ
# ?¨ê³„: 1?¨ê³„ - ?°ì´???„ì²˜ë¦?
# ?¤ëª…: ?´ë?ì§€ ?Œì „, ?´ë™, ?¸ì´ì¦?ì¶”ê?ë¥??µí•œ ?™ìŠµ ?°ì´???•ì¥
# ?‘ì„±?? 2026.05.13
# ================================================
import albumentations as A
import cv2
import os
import glob

def augment_images(input_dir, output_dir, n_aug=2):
    print(f"--- Data Augmentation ?œì‘: {input_dir} ---")
    
    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1),
    ])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    
    count = 0
    for img_path in images:
        image = cv2.imread(img_path)
        fname = os.path.basename(img_path)
        
        # Original save (optional, if moving to new dir)
        # cv2.imwrite(os.path.join(output_dir, fname), image)
        
        for i in range(n_aug):
            augmented = transform(image=image)['image']
            out_name = f"aug_{i}_{fname}"
            cv2.imwrite(os.path.join(output_dir, out_name), augmented)
            count += 1
            
    print(f"- ?ì„±??ì¦ê°• ?´ë?ì§€: {count}??)
    print(f"--- Data Augmentation ?„ë£Œ ---\n")

# ?¤í–‰ ?ˆì‹œ
# augment_images('raw_images/', 'augmented_images/')
