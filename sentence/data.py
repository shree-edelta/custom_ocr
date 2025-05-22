import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def get_image_path(base_dir, img_id):
    # base_dir should be the root to "words/" folder
    parts = img_id.split("-")
    subdir = os.path.join(base_dir, parts[0], f"{parts[0]}-{parts[1]}")
    filename = f"{img_id}.png"
    return os.path.join(subdir, filename)

data = []
with open("iam_words/words.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.strip().split()
        # print(parts)
        if len(parts) >= 7:
            path = parts[0]
            path = get_image_path("iam_words/words",path)
            word = ' '.join(parts[8:])
            # print(word)
            data.append([path, word])
            print(parts[8:])
            
df1 = pd.DataFrame(data, columns=["path", "word"])
df1.to_csv("data.csv",index = None)
print(df1.head(20))

df = pd.read_csv("data.csv")

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("valid.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("process complete.")

