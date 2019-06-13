# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:53:12 2019

@author: ashok.swarna
"""

from app_settings import read_config
from Utility import load_data
from Utility import load_conditional_data
from os import path
from pmdarima.arima import auto_arima
from sklearn.externals import joblib

def evaluate_models(data):
    
    arima = auto_arima(data,error_action='ignore', 
                       trace= False ,seasonal=True, m=7, suppress_warnings=True) 
                       
    return arima

def save_model_to_disk(arima,file_path):

# Pickle it
    joblib.dump(arima, file_path, compress=3)
    print('%s saved' % file_path)

def train():
    """
        Trains ARIMA model as per Region, State $ sku 
        & selects the best model and saves it
    """
    app_settings = read_config()
    data_path = app_settings['data_path']
    file  = app_settings['file']
    file_path = path.join(data_path, file)
    print(file_path) 
#Load the data set
    df = load_data(file_path)

#Extract the unique codes for filtering the over all data    
    sku_group = df.groupby('Product_SKU', as_index=False)
    sku_list = sku_group.groups.keys()

    region_group = df.groupby('Region', as_index=False)
    region_list = region_group.groups.keys()

    state_group = df.groupby('State', as_index=False)
    state_list = state_group.groups.keys()
    
#Loops for getting the conditional data  and fitting the model
    
    for Region in region_list:
        for state in state_list:
            for sku in sku_list:
                print ('Current filtering condition :', Region, state, sku )

# load the conditional data for each condition
                
                conditional_df = load_conditional_data(Region, state, sku, df)
                
                data = conditional_df.Sales.reset_index(drop=True)
                train, test = data[:330], data[330:365]
# #################Fit model with some validation (cv) samples ##############
                
                arima = evaluate_models(train)
                
# Save model to disk
                result_path = app_settings['result_path']
                model_file_name = Region + state + sku + ".Pickle" 
                model_file_path = path.join(result_path, model_file_name)
                save_model_to_disk(arima, model_file_path)
                
# #################saving the model as pickel file ##############
                
                
    print('____________________________________________________________________________________________')
    print('Training completed')
                
def main(): 
    train()


if __name__ == '__main__':
    main()
    