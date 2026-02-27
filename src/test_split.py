from dataset import load_image_paths_and_labels, create_split, save_split

ROOT = "C:/Users/Kwame Boateng/Documents/GitHub/AI_In_Medicine_Project_1/data/caltech-101"          #Adjust      
OUT_DIR = "splits"                #split files directory
EXCLUDE = {"BACKGROUND_Google"}   # exclude negative class

paths, labels, label2idx = load_image_paths_and_labels(ROOT, exclude=EXCLUDE)

train_paths, test_paths, train_labels, test_labels = create_split(
    paths, labels, test_size=0.30, seed=42
)

save_split(
    train_paths, test_paths, train_labels, test_labels,
    label2idx=label2idx, out_dir=OUT_DIR, seed=42, test_size=0.30
)

print("Saved split to:", OUT_DIR)
print("Train:", len(train_paths), "Test:", len(test_paths))