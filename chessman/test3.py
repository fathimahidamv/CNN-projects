from tensorflow.keras.preprocessing import image
from keras.models import load_model
import numpy as np
model=load_model('weight3.h5')
# testing the model
def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    result = model.predict(x= test_image)
    print(result)
    if np.argmax(result)  == 0:
      prediction = 'Bishop'
    elif np.argmax(result)  == 1:
      prediction = 'King'
    elif np.argmax(result)  == 2:
      prediction = 'Knight'
    elif np.argmax(result)  == 3:
      prediction = 'Pawn'
    elif np.argmax(result)  == 4:
      prediction = 'Queen'

    else:
      prediction = 'Rook'
    return prediction

print(testing_image(r'C:\Users\fathi\OneDrive\Desktop\CNN_PROJECTS\chessman\data\Knight\00000017.jpg'))

# from sklearn.metrics import  confusion_matrix

# Y_pred = model.predict_generator(xtest)

# y_pred = np.argmax(Y_pred, axis=1)

# print('Confusion Matrix')

# c=confusion_matrix(ytest, Y_pred)
# #cm = confusion_matrix(np.where(ytest), Y_pred)

