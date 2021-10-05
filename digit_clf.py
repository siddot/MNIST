import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

digits_df = load_digits()

X = digits_df.data
y = digits_df.target

_, axes = plt.subplots(nrows=1, ncols=10, figsize=(30,20))

for ax, image, label in zip(axes, digits_df.images, digits_df.target):
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' %label)

n_samples = len(digits_df.images)
data = digits_df.images.reshape((n_samples, -1))

clf = SVC(gamma=0.001) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)

# predictions
predicted = clf.predict(X_test)
print(X_test.shape)
_, axes = plt.subplots(nrows=1, ncols=100, figsize=(30, 20))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

print('Predicted: ',predicted)
print('Accuracy: ',classification_report(y_test, predicted))
