# Chicago Air Quality Sensor Network Analysis

This project analyzes air quality data from Chicago's sensor network using open source data from the Socrata API.

## Setup Instructions

## Deploy to Vercel (Flask)

This folder is now configured for Vercel serverless deployment.

### Included deployment files

- [api/index.py](api/index.py): serverless entrypoint that exposes `app`
- [vercel.json](vercel.json): routes all paths to the Flask app
- [.vercelignore](.vercelignore): excludes local venv/notebooks from upload

### Deploy steps

1. Install and log in to Vercel CLI:
   - `npm i -g vercel`
   - `npx vercel login`
2. From this folder, deploy:
   - `npx vercel`
3. For production deploy:
   - `npx vercel --prod`

### Environment variables

- Optional: `SOCRATA_APP_TOKEN` (recommended for higher API limits)
- If omitted, the app will still run using public API access and/or cached data.

### 1. Virtual Environment

The virtual environment `SensorsNetwork_Venv` has been created. To activate it:

**Windows PowerShell:**
```powershell
.\SensorsNetwork_Venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
SensorsNetwork_Venv\Scripts\activate.bat
```

### 2. Install Dependencies

All required packages are already installed. If you need to reinstall:

```bash
pip install -r requirements.txt
```

### 3. Configure Jupyter Kernel

To use this virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=SensorsNetwork_Venv --display-name "Python (SensorsNetwork)"
```

Then in Jupyter, select the kernel "Python (SensorsNetwork)" for the notebook.

## Project Structure

- `ChicagoAQI.ipynb` - Main analysis notebook
- `requirements.txt` - Python package dependencies
- `chicago_aqi_sensors_map.html` - Interactive sensor map (generated after running notebook)
- `chicago_aqi_interactive_timeseries.html` - Animated time-series visualization (generated after running notebook)

## Data Source

- **API**: Chicago Open Data Portal (Socrata)
- **Dataset ID**: xfya-dxtq
- **Reference**: https://dev.socrata.com/foundry/data.cityofchicago.org/xfya-dxtq

## Features

1. ✅ Load data from Socrata API
2. ✅ Display dataset head and basic statistics
3. ✅ Create interactive map with clickable sensors
4. ✅ **Time series data preprocessing** with time binning
5. ✅ **IDW (Inverse Distance Weighting) interpolation**
6. ✅ **Kriging interpolation**
7. ✅ **Interactive animated time-series visualization** with play/pause controls
8. ✅ **Statistics panel** (mean, std, min, max, quantiles)
9. ✅ **Quantile-based region filtering**
10. ✅ **Standalone HTML export** for sharing
11. 🔄 Network analysis (next step)
12. 🔄 Census data integration for socioeconomic analysis (next step)
13. 🔄 Spatiotemporal Graph Neural Network using tsl (next step)

## Usage

### Running the Notebook

1. Activate the virtual environment:
   ```powershell
   .\SensorsNetwork_Venv\Scripts\Activate.ps1
   ```

2. Open `ChicagoAQI.ipynb` in Jupyter

3. Run all cells to:
   - Load data from the API
   - Create static sensor map
   - Generate time-series preprocessing
   - Create interactive animated visualization with interpolation

### Interactive Features

- **Time Animation**: Use the play button or slider to animate through time
- **Interpolation Methods**: Switch between IDW and Kriging
- **Quantile Filtering**: Select quantile ranges to highlight specific air quality regions
- **Statistics Panel**: View real-time statistics (mean, std, min, max, quantiles)

### Output Files

- `chicago_aqi_sensors_map.html` - Static map with sensor locations
- `chicago_aqi_interactive_timeseries.html` - Standalone interactive animation (can be shared)

## Next Steps

1. Integrate census data for socioeconomic comparison
2. Add data quality metrics visualization
3. Export to Quarto/MyST format for document generation
4. Analyze sensor network structure
5. Implement spatiotemporal GNN using the tsl library
