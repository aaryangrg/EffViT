import csv

csv_dir  = "./efficientvit/mini_csv_files/allimages.csv"
# 600 samples / class x 100 
images = {}
with open(csv_dir) as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    next(csv_reader, None)
    for row in csv_reader:
        if len(row) == 2 and row[1] in images.keys():
            images[row[1]].append(row[0])
        else:
            if len(row) == 2 :
                images[row[1]] = [row[0]]

images = list(images.keys())

with open("./efficientvit/mini_csv_files/class_list.txt", 'w') as f:
        for cls in images:
            f.write(cls + '\n')
