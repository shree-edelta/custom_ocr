from PIL import Image
import os
import cv2
import csv

def convert_to_black_white_pil(image_path, threshold=128):
    img = Image.open(image_path).convert("L")  
    bw = img.point(lambda x: 255 if x > threshold else 0, mode='1')  
    return bw

# f_image_folder = 'images/train'
f_image_folder = 'images/valid'
s_image_folder = 'output'
output_csv = 'valid.csv'


for filename in os.listdir(f_image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        f_image_path = os.path.join(f_image_folder, filename)
        s_image_path = os.path.join(s_image_folder, filename)
        
        # image = cv2.imread(f_image_path)

        processed_result = convert_to_black_white_pil(f_image_path)
        processed_result.save(s_image_path)
       
        print(processed_result)
        
# labels = ['NAME','Hobbies','s.', 'BUTTON', 'ALWAYS','for all','NOT','Satani','Shree','Name'] 

labels = ['Button','CONTACT']

image_filenames = sorted([
    f for f in os.listdir(f_image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

assert len(labels) == len(image_filenames), "Mismatch: labels and images count differ!"

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])

    for filename, label in zip(image_filenames, labels):
        writer.writerow([filename, label])
    
