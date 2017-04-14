import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


def loadTrainData():

    x1 = np.load('./gendistx.npy'); x1 = np.divide(x1, np.max(x1))
    x2 = np.load('./fakedistx.npy'); x2 = np.divide(x2, np.max(x2))

    y1 = np.load('./gendisty.npy'); y1 = np.divide(y1, np.max(y1))
    y2 = np.load('./fakedisty.npy'); y2 = np.divide(y2, np.max(y2))

    X1 = np.hstack((x1, x2))
    X2 = np.hstack((y1, y2))

    X = np.vstack((X1,X2)).T

    Y = np.array([np.ones((2000,1)), np.zeros((2000,1))]).reshape((4000,1))

    return X, Y

def loadTestData():

    X = np.loadtxt('./testDTW.txt', delimiter=' ')

    return X


def buildModel(X_train, Y_train):

#******************** MODEL BULIDING *********************************************************
#********** Tweak the parameters here for the best results ***********************************

    batch_size = 512    # Small batch size leads to an overfit
    n_epoch = 20

    model = Sequential()
    model.add(Dense(64, input_dim=2, activation='relu'))
    model.add(Dropout(0.2))                     # No clue why dropout is reqd. Dense == fully connected
    model.add(Dense(32, activation='relu'))     # Observed that 'relu' non linearity gives best results
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))   # softmax here classifies everything to 1.0. Hence sigmoid

    # 3 layer fully connected network is bult now

    model.compile(loss='binary_crossentropy',optimizer='rmsprop',
              metrics=['accuracy'])

    # put verbose = 1 to see the model train accuracy printed on stdout for each epoch
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1)

    return model


if __name__ == "__main__":

    X_train, Y_train = loadTrainData()
    X_test = loadTestData()

    model = buildModel(X_train, Y_train)

    prediction = []
    for i in range(len(X_test)):
        prediction.append(np.ndarray.tolist(model.predict(np.array([X_test[i]])))[0][0])


    # model.predict returns a number between 0 and 1. If this is > some threshold, we classify it as
    # genuine, otherwise fake. This threshold could be the mean of the predicted values, median of the
    # values, or anything. Experiment with this and observe the results

    thresh = np.mean(np.array([prediction]))
    # thresh = np.median(np.array([prediction]))


    # Test data is odered like this: [Gen Gen Fake Fake Gen Gen Fake Fake .... ]. ===> OR IS IT??? DOUBT!!
    # So flag remains 1 for first 2 samples, 0 for next 2 samples and so on.
    # A counter counts from 1 to 2 and resets the flag as it becomes >2 (it resets itself to 1 after this)
    # Result is appended to a list: 1 for correct prediction, 0 for a miss.

    result = []; counter = 1; flag = 1
    for i in range(len(prediction)):

        if flag == 1 and prediction[i] > thresh:
            result.append(1)
        elif flag == 0 and prediction[i] < thresh:
            result.append(1)
        else:
            result.append(0)

        counter+=1
        if counter>2:
            flag = not flag
            counter = 1

    # print prediction
    # print result

    accuracy = (float(result.count(1))/len(result))*100
    print accuracy
