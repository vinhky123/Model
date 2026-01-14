# 🎨 Time Series Forecasting Demo

Beautiful web interface for deep learning time series forecasting models.

## ✨ Features

- 🚀 Load and run inference with trained models
- 📊 Interactive visualization with Chart.js
- 📈 Compare Input, Prediction, and Ground Truth
- 🎯 Real-time metrics (MAE, MSE, RMSE)
- 🎨 Modern, responsive UI design
- 🔄 Easy model switching

## 📁 Project Structure

```
demo/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # HTML template
├── static/
│   ├── style.css         # Styling
│   └── script.js         # Frontend logic
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd demo
pip install -r requirements.txt
```

### 2. Prepare Model Checkpoints

Create a `params/` directory in the project root and place your trained model checkpoints there:

```
params/
└── long_term_forecast_weather_96_96_TimeStar_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/
    └── checkpoint.pth
```

Or copy from existing checkpoints:

```bash
mkdir -p ../params
cp -r ../checkpoints/long_term_forecast_* ../params/
```

### 3. Prepare Dataset

Make sure ETTm2.csv is in the correct location:

```bash
# Default location:
dataset/ETT-small/ETTm2.csv
```

### 4. Run Demo Server

```bash
python app.py
```

Server will start at: `http://localhost:5000`

## 🎮 How to Use

1. **Open browser** → Navigate to `http://localhost:5000`

2. **Select Model** → Choose from dropdown (e.g., TimeStar, TimeXer, etc.)

3. **Load Model** → Click "Load Model" button
   - Wait for model to load (~5-10 seconds)
   - Model info will appear on the right

4. **Configure Inference**:
   - **Sample Index**: Select which test sample to use (0 to N-1)
   - **Channel**: Choose which feature channel to visualize (0-6 for ETTm2)

5. **Run Inference** → Click "Run Inference"
   - View visualization with 3 lines:
     - 🔵 **Blue**: Input sequence (96 timesteps)
     - 🟢 **Green**: Model prediction (96 timesteps)
     - 🔴 **Red (dashed)**: Ground truth (96 timesteps)
   - Check metrics (MAE, MSE, RMSE) on the right

6. **Experiment**:
   - Try different samples
   - Switch channels
   - Compare different models

## 🎨 UI Components

### Control Panel
- Model selection dropdown
- Sample index slider
- Channel selector
- Action buttons

### Visualization
- Interactive Chart.js line chart
- Hover to see values
- Zoom and pan support
- Legend toggle

### Info Cards
- Model information
- Performance metrics
- Real-time status

## 🔧 Configuration

### Change Dataset

Edit `demo/app.py`:

```python
class InferenceArgs:
    def __init__(self):
        self.data = 'ETTm2'  # Change to ETTh1, ETTh2, ETTm1, etc.
        self.root_path = './dataset/ETT-small/'
        self.data_path = 'ETTm2.csv'  # Change accordingly
```

### Change Model Parameters

Edit default parameters in `InferenceArgs` class:

```python
self.seq_len = 96      # Input sequence length
self.pred_len = 96     # Prediction length
self.d_model = 512     # Model dimension
# ... other params
```

### Add Custom Models

1. Make sure model is imported in `app.py`:
   ```python
   from models import YourModel
   ```

2. Place checkpoint in `params/` directory

3. Model will automatically appear in dropdown

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Model Not Found
```bash
# Check checkpoint directory structure
ls -la params/

# Make sure checkpoint.pth exists
find params/ -name "checkpoint.pth"
```

### Import Error
```bash
# Install missing dependencies
pip install -r requirements.txt

# Or manually:
pip install flask torch numpy pandas
```

### Data Not Found
```bash
# Verify dataset path
ls -la dataset/ETT-small/ETTm2.csv

# Update path in app.py if needed
```

## 📊 Example Output

**Visualization:**
```
Input (blue) ──────────────────▶
                              ▼
Prediction (green) ─────────────────────▶
Ground Truth (red, dashed) ─────────────▶
```

**Metrics:**
```
MAE:  0.1234
MSE:  0.0567
RMSE: 0.2381
```

## 🚀 Advanced Usage

### Run on Different Host/Port

```bash
python app.py --host 0.0.0.0 --port 8080
```

### Production Deployment

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

## 🎯 API Endpoints

- `GET /` - Main page
- `GET /api/models` - List available models
- `POST /api/load_model` - Load a specific model
- `POST /api/inference` - Run inference
- `GET /api/info` - Get current model info

## 📝 Notes

- First inference may be slow due to model loading
- Subsequent inferences are faster
- CPU inference is used by default (change in `InferenceArgs`)
- Chart supports up to 1000 points smoothly

## 🌟 Features to Add (Future)

- [ ] Multiple sample comparison
- [ ] Export predictions to CSV
- [ ] Upload custom data
- [ ] Real-time inference
- [ ] Model comparison mode
- [ ] Download visualization as image

---

**Enjoy the demo! 🎉**


