import csv
import json

def write_json(source, target, image_path):
    with open('output.json', 'a') as json_file:
        json.dump({"image": image_path, "source": source, "target": target}, json_file)
        json_file.write('\n')

# Clear previous content of output file
open('output.json', 'w').close()

with open('metadata.csv', newline='') as csvfile:
    csv_reader = csv.DictReader(csvfile, delimiter='\t')
    for row in csv_reader:
        image_info = row['image_id,domain,split,image_path']
        image_path = image_info.split(',')[-1]  # 获取最后一个逗号后的内容
        if 'trainA' in image_path:
            write_json("monet", "photo", image_path)
        elif 'trainB' in image_path:
            write_json("photo", "monet", image_path)

print("JSON file written successfully!")
