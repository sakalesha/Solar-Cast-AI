import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models import build_baseline_lstm, build_attention_lstm

def load_processed_data(city="la", data_dir="processed_data"):
    """
    Loads the processed tensors for a specific city.
    """
    file_path = os.path.join(data_dir, f"{city}_data.npz")
    data = np.load(file_path)
    return data['X'], data['y']

def train_model(model_name="baseline", city="la"):
    """
    Trains the specified model on the city's dataset.
    Returns the final test metrics.
    """
    print(f"\n--- Training {model_name.upper()} Model on {city.upper()} data ---")
    
    # 1. Load Data
    X, y = load_processed_data(city)
    
    # 2. Split Data (75% Train, 12.5% Val, 12.5% Test)
    n = len(X)
    train_end = int(n * 0.75)
    val_end = int(n * 0.875)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Val samples:   {len(X_val)}")
    
    # 3. Build Model
    if model_name == "baseline":
        model = build_baseline_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    elif model_name == "attention":
        model = build_attention_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    else:
        raise ValueError("Invalid model name")
    
    # 4. Callbacks
    # Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
        
    checkpoint_path = f"models/{model_name}_{city}_best.keras"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    ]
    
    # 5. Fit
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Final Evaluation on Test Set
    print(f"\nEvaluating {model_name} on Test set...")
    metrics = model.evaluate(X_test, y_test, verbose=0)
    # model.metrics_names provides the name of columns
    # Usually: ['loss', 'mae', 'root_mean_squared_error']
    rmse = metrics[2]
    mae = metrics[1]
    
    print(f"   Test RMSE: {rmse:.4f}")
    print(f"   Test MAE:  {mae:.4f}")
    
    return {"rmse": rmse, "mae": mae}

def run_all_experiments():
    """
    Iterates through all cities and models, saving results to results.csv.
    """
    cities = ["la", "bengaluru", "delhi", "chennai"]
    models = ["baseline", "attention"]
    
    results_file = "results.csv"
    file_exists = os.path.isfile(results_file)
    
    with open(results_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["city", "model", "rmse", "mae"])
        if not file_exists:
            writer.writeheader()
            
        for city in cities:
            for model_type in models:
                try:
                    metrics = train_model(model_name=model_type, city=city)
                    writer.writerow({
                        "city": city,
                        "model": model_type,
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"]
                    })
                    f.flush() # Ensure it writes during the long run
                except Exception as e:
                    print(f"Error training {model_type} on {city}: {e}")

if __name__ == "__main__":
    run_all_experiments()
    print("\nTraining Pipeline Complete. Check results.csv for summary.")
