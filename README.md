# ⚡ AI-Enhanced Atmospheric Electricity Harvesting Dashboard

A real-time monitoring and intelligent decision-making system for atmospheric energy harvesting systems, built with Streamlit and machine learning.

## 🚀 Features

- **Real-time MQTT Integration** - Connect to IoT devices and sensors
- **AI-Powered Anomaly Detection** - Uses Isolation Forest for detecting system anomalies
- **Live Data Visualization** - Interactive charts with Plotly
- **Intelligent Decision Making** - Automated harvesting mode recommendations
- **Data Export** - Download CSV files for analysis
- **Simulation Tools** - Generate test data with anomaly injection
- **Modern UI** - Clean, responsive interface with proper spacing

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for MQTT broker access

### Dependencies
All dependencies are listed in `requirements.txt`:

```bash
# Core dependencies
streamlit>=1.28.0      # Web framework
pandas>=2.0.0          # Data processing
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.3.0    # Machine learning
joblib>=1.3.0          # Model serialization
plotly>=5.15.0         # Data visualization
paho-mqtt>=1.6.1       # MQTT communication
```

## 🛠️ Installation

### Option 1: Quick Start
```bash
# Clone or download the project
cd "path/to/project"

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

### Option 2: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

### Option 3: Minimal Installation
```bash
# For production deployment
pip install -r requirements-minimal.txt
streamlit run dashboard.py
```

## 🎯 Usage

### 1. Start the Dashboard
```bash
streamlit run dashboard.py
```

### 2. Configure the System
1. **Load Model & Scaler** (if available)
   - Set model path: `phase3_predict_model.pkl`
   - Set scaler path: `phase3_predict_scaler.pkl`
   - Click "Load Model"

2. **Connect MQTT** (optional)
   - Host: `broker.hivemq.com` (default)
   - Port: `1883`
   - Topic: `aiharvest/iot`
   - Click "Connect MQTT"

3. **Upload Data** (alternative)
   - Use file uploader for CSV/JSON files
   - Or use the demo simulator

### 3. Monitor & Analyze
- View real-time metrics and KPIs
- Analyze anomaly detection results
- Monitor harvesting mode recommendations
- Export data for further analysis

## 📊 Data Format

### Expected MQTT JSON Payload
```json
{
  "timestamp": "2025-08-14T10:00:00Z",
  "voltage_V": 3.7,
  "current_A": 0.05,
  "power_W": 0.185,
  "energy_usage_J": 11.1,
  "temperature_C": 29.5,
  "humidity_%": 65,
  "field_strength_V_m": 190,
  "storage_level_%": 55,
  "sensor_active": 1
}
```

## Input Parameters

```json
{
   voltage_V
   current_A
   power_W (optional, computed)
   energy_usage_J (optional, computed)
   temperature_C
   humidity_%
   field_strength_V_m
   storage_level_%
   sensor_active
}
```

### CSV Format
The dashboard expects columns in this order:
```
timestamp,voltage_V,current_A,power_W,energy_usage_J,temperature_C,humidity_%,field_strength_V_m,storage_level_%,sensor_active
```

## 🤖 AI Decision Logic

The system uses the following logic for harvesting mode recommendations:

1. **Anomaly Detected** → `diagnostic` mode, `low_power` sensors
2. **High Storage (>70%) & High Field (>300V/m)** → `max` harvesting, `full` sensors
3. **Low Storage (<30%) & Low Field (<200V/m)** → `normal` harvesting, `low_power` sensors
4. **Default** → `normal` harvesting, `normal` sensors

## 🔌 IoT Integration

### Arduino/ESP32 Example
```cpp
#include <PubSubClient.h>
#include <WiFi.h>
#include <ArduinoJson.h>

// MQTT Configuration
const char* mqtt_server = "broker.hivemq.com";
const int mqtt_port = 1883;
const char* mqtt_topic = "aiharvest/iot";

void publishData() {
  StaticJsonDocument<200> doc;
  
  doc["timestamp"] = getTimestamp();  // ISO 8601 format
  doc["voltage_V"] = readVoltage();
  doc["current_A"] = readCurrent();
  doc["temperature_C"] = readTemperature();
  doc["humidity_%"] = readHumidity();
  doc["field_strength_V_m"] = readFieldStrength();
  doc["storage_level_%"] = readStorageLevel();
  doc["sensor_active"] = 1;
  
  String payload;
  serializeJson(doc, payload);
  
  client.publish(mqtt_topic, payload.c_str());
}
```

### Python Publisher (Raspberry Pi)
```python
import json
import time
import random
from datetime import datetime
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)

while True:
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "voltage_V": round(random.uniform(3.3, 4.2), 2),
        "current_A": round(random.uniform(0.02, 0.1), 3),
        "temperature_C": round(random.uniform(25, 35), 1),
        "humidity_%": round(random.uniform(40, 80), 1),
        "field_strength_V_m": round(random.uniform(150, 260), 1),
        "storage_level_%": round(random.uniform(20, 100), 1),
        "sensor_active": 1,
    }
    
    client.publish("aiharvest/iot", json.dumps(payload), qos=0, retain=False)
    time.sleep(5)
```

## 🏗️ Project Structure

```
project/
├── dashboard.py              # Main dashboard application
├── requirements.txt          # Full dependencies
├── requirements-minimal.txt  # Minimal dependencies
├── README.md                # This file
├── phase3_predict_model.pkl # AI model (if available)
├── phase3_predict_scaler.pkl # Data scaler (if available)
├── iot_anomaly_dataset.json # Sample dataset
├── new_iot_data.csv         # Sample CSV data
├── output.csv               # Output data
└── predict.py               # Prediction script
└── arduino.ino              # Arduino/ESP32 code
└── iot circuit diagram      # circuit diagram
```

## 🔧 Configuration

### Environment Variables (Optional)
```bash
# MQTT Configuration
MQTT_HOST=broker.hivemq.com
MQTT_PORT=1883
MQTT_TOPIC=aiharvest/iot

# Model Configuration
MODEL_PATH=phase3_predict_model.pkl
SCALER_PATH=phase3_predict_scaler.pkl
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"

[browser]
gatherUsageStats = false
```

## 🚀 Deployment

### Local Development
```bash
streamlit run dashboard.py --server.port 8501
```

### Production Deployment
```bash
# Using Docker (recommended)
docker build -t ai-harvesting-dashboard .
docker run -p 8501:8501 ai-harvesting-dashboard

# Using systemd (Linux)
sudo systemctl enable streamlit-dashboard
sudo systemctl start streamlit-dashboard
```

### Cloud Deployment
- **Heroku**: Use `Procfile` with `streamlit run dashboard.py`
- **AWS**: Deploy on EC2 with nginx reverse proxy
- **Google Cloud**: Use Cloud Run or Compute Engine
- **Azure**: Deploy on App Service or Container Instances

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=dashboard
```

### Manual Testing
1. Start the dashboard
2. Use the demo simulator to generate test data
3. Verify anomaly detection works
4. Test MQTT connectivity
5. Export data and verify format

## 📈 Performance

### Optimization Tips
- Use `buffer_limit` to control memory usage
- Adjust `refresh_sec` for update frequency
- Enable/disable auto-refresh as needed
- Use anomaly-only filtering for large datasets

### Monitoring
- Check memory usage in sidebar
- Monitor MQTT connection status
- Review log entries for errors
- Export data periodically for backup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

**MQTT Connection Failed**
- Check internet connection
- Verify broker host and port
- Ensure firewall allows MQTT traffic

**Model Loading Error**
- Verify model files exist
- Check file permissions
- Ensure scikit-learn version compatibility

**Memory Issues**
- Reduce buffer_limit
- Disable auto-refresh
- Restart the application

### Getting Help
- Check the logs in the sidebar
- Review the documentation
- Open an issue on GitHub
- Contact the development team

## 🔮 Future Enhancements

- [ ] Real-time alerts and notifications
- [ ] Advanced anomaly detection algorithms
- [ ] Mobile-responsive design
- [ ] Database integration for historical data
- [ ] API endpoints for external integrations
- [ ] Machine learning model training interface
- [ ] Multi-device support
- [ ] Advanced analytics and reporting

---

**Built with ❤️ for sustainable energy harvesting**

