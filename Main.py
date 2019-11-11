import numpy as np


class One_vs_all_fisher:
  def __init__(self, images, labels):
    self.images = images
    self.labels = labels
    self.classes = np.unique(lbls)
    self.train()

# Function to calculate the mean of the intended class and the mean of other classes
  def m(self, c1, images_c1, images_c2):
      m1= []
      for i in range(images_c1.shape[1]):
          m1.append(np.sum(images_c1[:,i])/images_c1[:,i].shape[0])

      m2=[]
      for i in range(images_c2.shape[1]):
          m2.append(np.sum(images_c2[:,i])/images_c2[:,i].shape[0])
      return np.asarray(m1).reshape(1,-1),  np.asarray(m2).reshape(1,-1)

# Function to calculate the covariance of the intended class and the mean of other classes
  def S(self, c1, images_c1, images_c2, means):
    term1 = images_c1 - means[0]

    term2 = images_c2 - means[1]
    return (term1.T).dot(term1), (term2.T).dot(term2)


  def train(self):
    # Dictionary that will contain W, W0, projected class values mean and standard deviatiom
    self.training_values = {}

    for i in self.classes:
      images_c1 =self.images[self.labels == i,:]
      images_c2 = self.images[self.labels != i,:]
      means = self.m(i, images_c1, images_c2)
      covariances = self.S(i, images_c1, images_c2, means)

      Sw = np.matrix(covariances[0] + covariances[1])
      w = np.linalg.pinv(Sw).dot((means[0]-means[1]).T)
      w0 = float(.5*(means[0]+means[1]).dot(w))

      y_class = np.asarray(np.dot(w.T,images_c1.T)).reshape(-1)
      y_mean = sum(y_class)/len(y_class)
      y_std = np.std(y_class)

      self.training_values[i] = w, w0, y_mean, y_std



  def classify(self, test_images):
    labels = np.empty((test_images.shape[0] ,) , dtype = int)
    zscores = np.empty((test_images.shape[0] ,) , dtype = int)
    y_all = []
    for i in self.classes:
      w = self.training_values[i][0]
      w0 = self.training_values[i][1]
      y_mean = self.training_values[i][2]
      y_std = self.training_values[i][3]
      y = np.asarray(np.dot(w.T,test_images.T)).reshape(-1)
      y_all.append(y)
      for image_index in range(len(y)):
        if labels[image_index] in self.classes:
          zscore = np.absolute((y[image_index] - y_mean)/y_std)
          if zscores[image_index] > zscore:
            labels[image_index] = i
            zscores[image_index] = zscore
        else:
          if y[image_index] > w0:
            zscore = np.absolute((y[image_index] - y_mean)/y_std)
            labels[image_index] = i
            zscores[image_index] = zscore
    for i in range(len(labels)):
      if labels[i] in self.classes:
        continue
      else:
        all_projections = np.asarray([y_all[int(ii)][i] for ii in self.classes])
        all_w0 = np.asarray([self.training_values[int(ii)][1] for ii in self.classes])
        distance = list((all_w0 - all_projections)/all_w0)
        labels[i] = distance.index(min(distance))
    return labels


import os
import imageio


home_path = os.getcwd()
path='./Train'
os.chdir(path)
Images=os.listdir()

Images1=sorted(Images, key=lambda t: int(os.path.splitext(t)[0])) # sort them ascendingly
ImagF=np.zeros((2400,784))  # All Images
for i in range(len(Images1)):
    ImagF1=imageio.imread(Images1[i])
    ImagF[i,:]=ImagF1.reshape((1,784))


os.chdir(home_path)
lbls=np.loadtxt("Training Labels.txt")
lbls_T=np.loadtxt("Test Labels.txt")
path='./Test'
os.chdir(path)
Test_Image=os.listdir()
Test_Image.pop()
Test_Image1=sorted(Test_Image, key=lambda t: int(os.path.splitext(t)[0]))
ImagF_T=np.zeros((200,784))
for i in range(len(Test_Image1)):
    Imag=imageio.imread(Test_Image1[i])
    ImagF_T[i,:]=Imag.reshape((1,784))


classifier = One_vs_all_fisher(ImagF, lbls)
# Calculate Predicted labels
t = classifier.classify(ImagF_T)


#Calculate Accuracy and Print it
Accuracy = 100*np.sum(t == lbls_T)/len(lbls_T)
print(f"Accuracy of predicted labels = {Accuracy}%")

def center(image):
   mask = image>95
   image = image[np.ix_(mask.any(1),mask.any(0))]
   shortest_dimension = min(image.shape)
   compensation = 14 - np.floor((shortest_dimension)//2)
   return np.pad(image, pad_width = int(compensation), mode = 'constant', constant_values = 0)


def crop(images,cropx,cropy):
  no_of_images = images.shape[0]
  images_cropped = np.empty((no_of_images , cropx * cropy) , dtype = int)

  for i in range(no_of_images):
    image = images[i,:].reshape(28,28)
    image = center(image)
    dimensions = image.shape
    startx = (dimensions[1] - cropx)//2
    starty = (dimensions[0]-cropy)//2
    cropped_image = image[starty:starty+cropy:,startx:startx+cropx].reshape(1, cropx * cropy)
    images_cropped[i,:] = cropped_image

  return images_cropped


import matplotlib.pylab as plt
ImagF_crop = crop(ImagF,23,23)
ImagF_T_crop = crop(ImagF_T,23,23)

%matplotlib inline

for i in range(1,10):
    axes1 = plt.subplot(1, 2, 1)
    axes1.imshow(ImagF[220+i*240,:].reshape(28,28))
    axes1.set_title('Before Pre-Processing')
    axes2 = plt.subplot(1, 2, 2)
    axes2.imshow(ImagF_crop[220+i*240,:].reshape(23,23))
    axes2.set_title('After Pre-Processing')
    plt.show()

classifier = One_vs_all_fisher(ImagF_crop, lbls)
t = classifier.classify(ImagF_T_crop)

Accuracy = 100*np.sum(t == lbls_T)/len(lbls_T)
print(f"Accuracy of predicted labels = {Accuracy}% with image 23 x 23")


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

cm = confusion_matrix(lbls_T, t)

print('Confusion Matrix: ')
print(cm)
print('\n')
print('Accuracy Score :',accuracy_score(lbls_T, t))
print('\n')
print('Report : ')
print(classification_report(lbls_T, t))



title = 'Confusion matrix'

cmap = plt.cm.Greens
classes = np.unique(lbls_T)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title=title,
       ylabel='True label',
       xlabel='Predicted label')

ax.margins(y = 5)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

os.chdir(home_path)
fig.savefig('Confusion Matrix.png')
