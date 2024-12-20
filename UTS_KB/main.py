import os
import numpy as np
import cv2 as cv
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt


# Load and preprocess images
TrainPath = r"TTD_A"
categories = [
    'TTD AvavSam',
    'TTD FAHRIL ANTONIO HANDE',
    'TTD Lyra Attallah Aurellia',
    'TTD MARVELLOUS DEMETRIUS MAIT',
    'TTD MUHAMMAD AL-FARABY MOIDADY',
    'TTD Qhiran',
    'TTD SOPI',
    'TTD Vicram',
    'TTD_aqilah',
    'TTD_NOVITA',
    'TTD BRIANT_CJ',
    'TTD Fitri Handayani',
    'TTD MAHRECZY ADITYA',
    'TTD MOHAMMAD RAIYAN',
    'TTD NAtsya labaso',
    'TTD RIZKA FILARDI TOLIZ',
    'TTD SUPARMAN',
    'TTD Yuyun',
    'TTD_Hasby Ashidiq',
    'TTD Fadhil Akmal Zakaria',
    'TTD Fransisca',
    'TTD MARSYA CIKITA',
    'TTD MUH HALIIM',
    'TTD PUTRI CASIOLA',
    'TTD Siti Nurvatika',
    'TTD VIAAAAAAAAA',
    'TTD Zaky Putra',
    'TTD_Muh. Mashaq Ramadhan. M'
]



data = []

for category in categories:
    label = categories.index(category)
    path = os.path.join(TrainPath, category)
    for img in os.listdir(path):
        imgPath = os.path.join(path, img)
        try:
            imgArray = cv.imread(imgPath)
            imgResized = cv.resize(imgArray, (50, 50))
            image = np.array(imgResized).flatten()
            data.append([image, label])
            print("success load image from ", imgPath)
        except Exception as e:
            print("error load image from ", imgPath)
            pass



# Save data to pickle file
with open('dataImage.pickle', 'wb') as pick:
    pickle.dump(data, pick)
print("success save data image")



# Load data from pickle file
with open('dataImage.pickle', 'rb') as pick:
    data = pickle.load(pick)




# Separate features and labels
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)



# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)



# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.1)




# Train the model
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)

print("Model training completed")


pick = open('model.sav', 'wb')
pickle.dump(model, pick)
pick.close()
print("Model saved")  


pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()
print("Model loaded")


categories = [
    'TTD AvavSam',
    'TTD FAHRIL ANTONIO HANDE',
    'TTD Lyra Attallah Aurellia',
    'TTD MARVELLOUS DEMETRIUS MAIT',
    'TTD MUHAMMAD AL-FARABY MOIDADY',
    'TTD Qhiran',
    'TTD SOPI',
    'TTD Vicram',
    'TTD_aqilah',
    'TTD_NOVITA',
    'TTD BRIANT_CJ',
    'TTD Fitri Handayani',
    'TTD MAHRECZY ADITYA',
    'TTD MOHAMMAD RAIYAN',
    'TTD NAtsya labaso',
    'TTD RIZKA FILARDI TOLIZ',
    'TTD SUPARMAN',
    'TTD Yuyun',
    'TTD_Hasby Ashidiq',
    'TTD Fadhil Akmal Zakaria',
    'TTD Fransisca',
    'TTD MARSYA CIKITA',
    'TTD MUH HALIIM',
    'TTD PUTRI CASIOLA',
    'TTD Siti Nurvatika',
    'TTD VIAAAAAAAAA',
    'TTD Zaky Putra',
    'TTD_Muh. Mashaq Ramadhan. M'
]
predict = model.predict(xtest)
accuracy = model.score(xtest, ytest)
print("Model accuracy: ", accuracy)
print("Prediction: ", categories[predict[0]])

# Reshape the image back to its original dimensions
mySign = xtest[0].reshape(50, 50, 3)
# Convert the image from BGR to RGB format
mySign = cv.cvtColor(mySign, cv.COLOR_BGR2RGB)
plt.imshow(mySign)
plt.show()
