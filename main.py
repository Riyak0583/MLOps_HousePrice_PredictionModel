from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import model_build
from src.model_evaluation import model_evaluate
from src.model_prediction import model_predict


def main():
    # Step1: Data Ingestion
    df = data_ingestion()
    print(df.columns)
   
    # Step2: Data Preprocessing
    X_train,X_test,y_train,y_test = data_preprocessing(df)
    print(X_train)
   
    # Step3: Model Building
    model = model_build(X_train,X_test,y_train,y_test)

    # Step4: Model Prediction
    y_pred = model_predict(model, X_test)

    # Step5 : Model Evaluation
    mae, mse, rmse, r2 = model_evaluate(y_test, y_pred)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

main()