import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def load_nsrdb_folder(folder_path, start_year=None, end_year=None):
    """
    Loads CSV files in a folder, optionally filtering by year range.
    NSRDB format: lines 0-1 metadata/units, line 2 column names.
    Filename expected to end with _YYYY.csv
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if start_year and end_year:
        filtered_files = []
        for f in all_files:
            try:
                # Extract year from filename (e.g., ..._2016.csv)
                year = int(f.split('_')[-1].split('.')[0])
                if start_year <= year <= end_year:
                    filtered_files.append(f)
            except:
                continue
        all_files = filtered_files
    
    all_files.sort()
    
    data_frames = []
    
    column_mapping = {
        'GHI': 'ghi',
        'Clearsky GHI': 'ghi_clear',
        'Temperature': 'temp',
        'Relative Humidity': 'rh',
        'Wind Speed': 'wind_speed',
        'Pressure': 'pressure'
    }

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, skiprows=2)
        
        existing_cols = [col for col in column_mapping.keys() if col in df.columns]
        df = df[existing_cols]
        df = df.rename(columns=column_mapping)
        
        data_frames.append(df)
    
    if not data_frames:
        raise ValueError(f"No data found in {folder_path} for years {start_year}-{end_year}")
        
    full_df = pd.concat(data_frames, ignore_index=True)
    return full_df

def preprocess_solar_data(df):
    """
    Interpolates missing values and calculates Clearness Index (Kt).
    Final feature set: [ghi, temp, rh, wind_speed, pressure, kt]
    """
    # Replace outliers/invalid data (-999 often used in NREL for missing)
    df.replace(-999, np.nan, inplace=True)
    
    # Linear interpolation for missing values
    df = df.interpolate(method='linear', limit_direction='both')
    
    # Calculate Clearness Index (Kt = GHI / GHI_clear) 
    df['kt'] = 0.0
    mask = df['ghi_clear'] > 5  
    df.loc[mask, 'kt'] = df.loc[mask, 'ghi'] / df.loc[mask, 'ghi_clear']
    
    # Clip Kt to reasonable range [0, 1.2] 
    df['kt'] = np.clip(df['kt'], 0, 1.2)
    
    # Final feature selection in strict order
    cols = ['ghi', 'temp', 'rh', 'wind_speed', 'pressure', 'kt']
    df_final = df[cols]
    
    return df_final

def create_windowed_dataset(data, window_size=72):
    """
    Converts dataframe to (samples, window_size, features) tensors.
    Predicts GHI one hour ahead.
    """
    vals = data.values
    X, y = [], []
    
    for i in range(len(vals) - window_size):
        X.append(vals[i : i + window_size])
        y.append(vals[i + window_size, 0])  
        
    return np.array(X), np.array(y)

def save_dataset(X, y, filename, output_dir="processed_data"):
    """
    Saves the windowed tensors to disk.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, filename)
    np.savez_compressed(file_path, X=X, y=y)
    print(f"Saved to {file_path}")

if __name__ == "__main__":
    # Filter all cities to the same 2016-2020 window
    cities = {
        "la": ("Dataset", 2016, 2020),
        "bengaluru": ("NewDataset/Bengaluru", 2016, 2020),
        "delhi": ("NewDataset/Delhi", 2016, 2020),
        "chennai": ("NewDataset/Chennai", 2016, 2020)
    }
    
    print("Starting SolarCast AI Data Pipeline (Time-Matched 2016-2020)...\n")
    
    for city, (path, start, end) in cities.items():
        print(f"Processing {city.upper()} ({start}-{end})...")
        try:
            df = load_nsrdb_folder(path, start_year=start, end_year=end)
            processed_df = preprocess_solar_data(df)
            X, y = create_windowed_dataset(processed_df)
            
            print(f"   - Shape: {X.shape}")
            save_dataset(X, y, f"{city}_data.npz")
        except Exception as e:
            print(f"   - Error processing {city}: {e}")
            
    print("\nAll datasets processed and stored in 'processed_data/'.")
