import pandas as pd
def data_ingestion():
    '''
    * Title : "MLOps HousePrice_PredictionModel".
    * Independent Column[X] : Property_ID,Date_Sold,Property_Type,Location_Type,
                            Neighborhood_Quality,Condition,SqFt,Lot_Size_SqFt,
                            Bedrooms,Bathrooms,Floors,Year_Built,Garage_Spaces,
                            Has_Pool,Has_Basement,Has_Fireplace,Has_Central_AC,
                            Heating_Type,Distance_To_City_Center_Miles,School_Rating,
                            HOA_Fee_Monthly,Annual_Property_Tax,
    * Dependent Column[y]   : Price

    '''

    df = pd.read_csv(r"C:\MLOPs_HousePrice_PredictionModel\data\raw\house_price_prediction_dataset.csv")
    return df

