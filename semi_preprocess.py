from pickle import load
import pandas as pd

dict_features={'txt':'comprehensiveincomenetoftax','csho':'commonstocksharesauthorized',
                'act':'assetscurrent','ni':'netincomeloss','xint':'incometaxexpensebenefit',
                'lct':'liabilitiescurrent','at':'assets','cogs':'costofgoodsandservicessold',
                'ni':'netincomeloss','lt':'liabilities',
                'ap':'accountspayablecurrent','ppegt':'propertyplantandequipmentnet',
                'rect':'billedcontractreceivables','pstk':'preferredstockvalue'}
dict_X={'txt':0,'csho':0,
                'act':0,'ni':0,'xint':0,
                'lct':0,'at':0,'cogs':0,
                'ni':0,'lt':0,
                'ap':0,'ppegt':0,
                'rect':0,'pstk':0}

def process_data(data):
    scalar = load(open('scaler.pkl', 'rb'))
    for key,val in enumerate(zip(data.keys(),data.values())):
        if key in list(dict_features.keys()):
            dict_X[key]=val
    X=pd.DataFrame(dict_X)
    scaled_testing = scalar.transform(X)

    # scaled_test = scalar.fit_transform(test_data_df)

    scaled_testing_df = pd.DataFrame(data = scaled_testing, columns=X.columns)
    scaled_testing_df.fillna(0)

    
    return scaled_testing_df.values