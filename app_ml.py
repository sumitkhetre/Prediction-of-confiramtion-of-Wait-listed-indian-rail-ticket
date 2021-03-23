# the code to build the ML model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_model():
    # load the data
    df = pd.read_csv('Trai_Ticket.csv')
    # clean the data
    x = df.drop(['bookingStatus', 'createdAtDate', 'isActive', 'isChartPrepared', 'pnrNo', 'trainNo','status1Day','status2Days','status1Month','status1Week', 'updatedAt', 'label'], axis=1)

    y = df['label']

    # convert the categorical to numerical
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()

    x['from'] = label_encoder.fit_transform(x['from'])
    x['journeyDate'] = label_encoder.fit_transform(x['journeyDate'])
    x['to'] = label_encoder.fit_transform(x['to'])
    x['travelClass'] = label_encoder.fit_transform(x['travelClass'])

    # print(df.head(20))
    # print(df['region'].unique())

    # sex: 0: female, 1: male
    # smoker: 1: yes, 0: no
    # region: 3: southwest, 2: southeast, 1: northwest, 0: northeast



    # split the data into train and test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234567)

    # model training
    from sklearn.svm import SVC
    model = SVC(kernel='rbf')
    model.fit(x_train, y_train)

    # dump the model
    import pickle
    file = open('model_lr.pkl', 'wb')
    pickle.dump(model, file)
    file.close()

    def test_model():
        from sklearn.metrics import accuracy_score
        predictions = model.predict(x_test)
        print(predictions)
        print(f"accuracy = {model.score(x_test, y_test)}%")

    # test_model()

    # def create_regression_line():
    #     predictions = model.predict(x)
    #     plt.scatter(x['bmi'], y)
    #     plt.plot(x['bmi'], predictions, color="red")
    #
    #     plt.xlabel('Age')
    #     plt.ylabel('charges')
    #     plt.title("regression line for insurance")
    #     plt.tight_layout()
    #     plt.savefig('static/regression.png')
    #     plt.show()
    #
    # create_regression_line()



def predict(from_1 , to_1 ,  travelClass ,journeyDate):


    # date incoded by using inbuilt function of python
    j=journeyDate
    day = j.split("-")
    journeyDate=day[1]
    # load the model
    import pickle

    filename = 'model_lr.pkl'

    with open(filename, 'rb') as file:
        model = pickle.load(file)


    confirm = model.predict([[from_1, to_1 , travelClass , journeyDate]])
    print(confirm[0])

    return confirm[0]



if __name__ == '__main__':
    train_model()
    # predict(19, 0, 27.9, 0, 1, 3)


