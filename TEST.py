from tutorial_dataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[16]
image = item['jpg']
source = item['source']
target = item['txt']
print(image.shape)
print(source)
print(target)