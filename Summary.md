# SolarCast AI - Research & Implementation Summary

## 1. Research Gap Analysis
Based on the paper *"Hourly Solar Irradiance Forecasting Using Long Short Term Memory and Convolutional Neural Networks"* (2024), we identified a critical gap:
- **Gap:** The model treats all 72 historical timesteps with equal importance.
- **Proposed Solution:** Implement a **Temporal Self-Attention mechanism** to allow the model to focus on the most relevant past hours (e.g., the same hour from the previous day $t-24$).

## 2. Proposed Architecture (LSTM-Att)
Our improved model (Option A) uses the following sequence:
1. **Input:** (Batch, 72, 6) — 72 hours of lookback with 6 meteorological features.
2. **LSTM Layers:** Stacked LSTMs with 32, 16, and 8 units.
3. **Temporal Attention:** A custom layer to weight the 72 hidden states.
4. **Dense Layers:** 200 units → 100 units → 1 unit (GHI prediction).

## 3. Implementation Progress

### Phase 1: Data Pipeline (Completed)
- **Datasets Merged:** NSRDB data for Los Angeles (21+ years) and Indian stations (Bengaluru, Delhi, Chennai).
- **Corrected Geo-mapping:** Files for Delhi and Chennai were swapped to match correct coordinates. (Delhi files = (28.58, 77.45), Chennai files = (13.06, 80.25)).
- **Feature Engineering:** Calculated the **Clearness Index ($K_t$)** as $GHI / GHI_{clear}$.
- **Storage:** Processed tensors stored in `processed_data/*.npz`.

### Phase 2: Custom Architecture (Completed)
- **Status:** Created `src/attention_layer.py` and `src/models.py`.
- **Baseline Model:** Replicated the paper's best-performing 3-layer LSTMs.
- **Improved Model (LSTM-Att):** Integrated the custom `TemporalAttention` layer into the stacked architecture.
- **Verification:** Both models compiled successfully with MSE loss and are ready for training.

## 4. Project Evolution & Major Decisions

### [2026-04-09] Shift to Time-Matched Experimental Design
- **Decision:** Restricted the training/testing window for all cities (including LA) to **2016–2020**.
- **Rationale:** The Los Angeles dataset originally spanned 27 years, while Indian stations had only 5 years. Matching the years isolates the **geographic shift** as the only variable, preventing the model from gaining an unfair advantage through longer historical exposure.
- **Result:** LA dataset regenerated to match the ~44k samples of Indian stations.

## 5. Current File Map
- `src/data_loader.py`: Merging, cleaning, and windowing (includes year filtering).
- `src/attention_layer.py`: Custom Keras layer for Temporal Attention.
- `src/models.py`: Definitions for Baseline and Attention models.
- `processed_data/`: Persistent binary storage for processed training data.

---
*Last Updated: 2026-04-09*
