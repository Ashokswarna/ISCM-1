# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:52:15 2019

@author: ashok.swarna
"""

import pandas as pd


def load_data(file_path):
    
    df = pd.read_excel(file_path)
    cols = ['Order_date', 'Region', 'State', 'Product SKU','Sales']
    df = df[cols].reset_index(drop=True)
    df = df.rename(columns={'Product SKU': 'Product_SKU'})
    
    return df

def load_conditional_data(Region, state, sku, df):
    
    df1 = df[df['Region'].isin ([Region]) & df['State'].isin([state]) &
             df['Product_SKU'].isin([sku]) ].reset_index(drop=True)
    
    df1.sort_values('Order_date')
    
    return df1
    
