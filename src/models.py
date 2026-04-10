import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, Dropout
from src.attention_layer import TemporalAttention

def build_baseline_lstm(input_shape=(72, 6)):
    """
    Replicates the Vanilla Stacked LSTM from the research paper.
    Architecture: LSTM(32) -> LSTM(16) -> LSTM(8) -> Dense(200) -> Dense(100) -> Dense(1)
    """
    inputs = Input(shape=input_shape)
    
    # Paper uses 3 LSTM layers
    x = LSTM(32, return_sequences=True)(inputs)
    x = LSTM(16, return_sequences=True)(x)
    x = LSTM(8, return_sequences=False)(x)
    
    x = Flatten()(x)
    
    # Dense Layers with ReLU
    x = Dense(200, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    
    # Output Layer (Linear for Regression)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Baseline_Stacked_LSTM")
    model.compile(optimizer='adam', loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    
    return model

def build_attention_lstm(input_shape=(72, 6)):
    """
    Proposed Improved Model: LSTM with Temporal Attention.
    Architecture: LSTM(32) -> LSTM(16) -> LSTM(8, return_seq=True) -> TemporalAttention -> Dense(200) -> Dense(100) -> Dense(1)
    """
    inputs = Input(shape=input_shape)
    
    # LSTMs
    x = LSTM(32, return_sequences=True)(inputs)
    x = LSTM(16, return_sequences=True)(x)
    # The last LSTM must return sequences for the Attention layer to work over timesteps
    x = LSTM(8, return_sequences=True)(x)
    
    # Custom Temporal Attention Layer
    x = TemporalAttention()(x)
    
    x = Flatten()(x)
    
    # Dense Layers
    x = Dense(200, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    
    # Output
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Temporal_Attention")
    model.compile(optimizer='adam', loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    
    return model

if __name__ == "__main__":
    # Sanity Check
    baseline = build_baseline_lstm()
    baseline.summary()
    
    print("\n" + "="*50 + "\n")
    
    att_model = build_attention_lstm()
    att_model.summary()
