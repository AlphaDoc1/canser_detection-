import os

def load_image_paths(root):
    image_paths = []
    labels = []
    
    for fold in os.listdir(root):
        fold_path = os.path.join(root, fold, fold)
        
        for cls in ["all", "hem"]:
            class_path = os.path.join(fold_path, cls)
            
            if not os.path.exists(class_path):
                continue
            
            for img in os.listdir(class_path):
                image_paths.append(os.path.join(class_path, img))
                labels.append(1 if cls == "all" else 0)
    
    return image_paths, labels
