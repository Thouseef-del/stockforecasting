import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_stock_data
from src.preprocess import preprocess_data
from src.train_model import train_model
from src.predict import predict_price
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset (use absolute path so app works from any current working directory)
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "etfs", "AAAU.csv")

data = load_stock_data(DATA_PATH)
print(f"Loaded {len(data)} rows | Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")

# Preprocess
X, y = preprocess_data(data)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = train_model(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation — MAE: {mae:.4f} | R² Score: {r2:.4f}")

# Predict next price from last known close
last_price = data['Close'].iloc[-1]
next_price = predict_price(model, last_price)

print(f"\nLast Closing Price : ${last_price:.2f}")
print(f"Predicted Next Price: ${next_price:.2f}")
