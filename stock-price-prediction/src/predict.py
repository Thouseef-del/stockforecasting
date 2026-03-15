def predict_price(model,last_price):
    
    prediction = model.predict([[last_price]])

    return prediction[0]