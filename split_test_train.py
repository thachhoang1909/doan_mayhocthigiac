import glob
from sklearn.model_selection import train_test_split

"""
# load feature
with open('features.pkl', 'rb') as feature_file:
    features = pickle.load(feature_file)

features = features.reshape((features.shape[0], features.shape[2]))
"""

# load image name
images = glob.glob('images\*\*.png')
images.sort()


# set labels for each image
targets = []
for image in images:
    targets.append(image.split("\\")[1])


# split to train/test set
X_train, X_test, Y_train, Y_test = train_test_split(images, targets, test_size=0.2)

with open('db\db2\\train.txt', mode='w') as train_file:
    for file in X_train:
        train_file.write(file)
        train_file.write('\n')

with open('db\db2\\lbtrain.txt', mode='w') as label_train_file:
    for file in Y_train:
        label_train_file.write(file)
        label_train_file.write('\n')

with open('db\db2\\test.txt', mode='w') as test_file:
    for file in X_test:
        test_file.write(file)
        test_file.write('\n')

with open('db\db2\\lbtest.txt', mode='w') as label_test_file:
    for file in Y_test:
        label_test_file.write(file)
        label_test_file.write('\n')

print("Done")
