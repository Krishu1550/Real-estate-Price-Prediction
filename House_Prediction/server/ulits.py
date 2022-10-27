
import numpy as np
import pickle
import json


model=None
data_columns=None
locations=None
def get_location_names():
    return locations

def load_data():
    global model
    global data_columns
    global locations
    print("Columns loading.. ")
    data_columns=json.load(open('../model/columns.json','r'))['data_columns']
    locations= data_columns[4:]
    print("Columns loaded")

    print("model loading")
    with open('../model/House_model.pkl','rb') as f:
        model=pickle.load(f)
    print("model loaded")
def predict_model(location,sqft,bath,balcony,bhk):
    loc_index=data_columns.index(location.lower())
    x=np.zeros(len(data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=balcony
    x[3]=bhk
    if loc_index>0:
        x[loc_index]=1

    return round(model.predict([x])[0],3)
if __name__=='__main__':
    load_data()
    get_location_names()
    print(predict_model('Whitefield',1000,2,3,2))
    print(predict_model('Vijayanagar',1000,2,3,2) )