def inference(X_test, y_test, model_info): 
    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(X_test)

    result = pd.DataFrame(X_test)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    print(result[:4])