import numpy as np
import pandas as pd

# Set reproducibility
np.random.seed(42)
samples = 20000

# --- Base factors ---
ambient_temp = np.random.normal(20, 10, samples).clip(-20, 50)
altitude = np.random.normal(8000, 3000, samples).clip(0, 15000)
flight_hours = np.random.exponential(2000, samples).clip(0, 5000)
last_service_cycles = np.random.exponential(300, samples).clip(0, 1000)

# --- Interdependent technical factors ---
pressure = 350 - (altitude / 1000) * np.random.uniform(5, 9) + np.random.normal(0, 10, samples)
pressure = pressure.clip(200, 400)

temperature = 600 + (pressure - 250) * 0.8 + ambient_temp * 0.2 + np.random.normal(0, 15, samples)
temperature = temperature.clip(500, 900)

vibration = (
    0.1 + 0.0001 * flight_hours + 0.0003 * last_service_cycles + np.random.normal(0, 0.05, samples)
)
vibration = vibration.clip(0, 1)

oil_quality = 100 - (flight_hours / 100) * np.random.uniform(0.5, 1.0) - vibration * 20
oil_quality = oil_quality.clip(50, 100)

# --- Compute realistic failure probability ---
# Components are more likely to fail with high temp, vibration, pressure, and low oil quality
failure_score = (
    0.002 * (temperature - 600)
    + 0.003 * (pressure - 250)
    + 2.5 * vibration
    - 0.015 * (oil_quality - 75)
    + 0.0002 * flight_hours
    + 0.0005 * last_service_cycles
)

# Convert to 0–1 range using logistic function
failure_probability = 1 / (1 + np.exp(-failure_score))
failure_probability = np.clip(failure_probability, 0, 1)

# --- Create DataFrame ---
data = pd.DataFrame({
    "Temperature (°C)": temperature.round(2),
    "Pressure (psi)": pressure.round(2),
    "Vibration (g)": vibration.round(3),
    "Oil Quality (%)": oil_quality.round(1),
    "Flight Hours": flight_hours.round(0).astype(int),
    "Ambient Temp (°C)": ambient_temp.round(1),
    "Altitude (ft)": altitude.round(0).astype(int),
    "Last Service Cycles": last_service_cycles.round(0).astype(int),
    "Failure Probability": failure_probability.round(4)
})

# Save to CSV
data.to_csv("aircraft_component_data.csv", index=False)

print(data.head())
