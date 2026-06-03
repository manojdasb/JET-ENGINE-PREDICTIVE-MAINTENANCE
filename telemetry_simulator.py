# telemetry_generator.py
import pandas as pd
import time

class TelemetryStreamer:
    def __init__(self, file_path="simulated_test_bank.csv"):
        self.df = pd.read_csv(file_path)
        
    def get_engine_stream(self, engine_id):
        """Yields single-row telemetric snapshots sequentially to act like a live generator."""
        engine_data = self.df[self.df['engine_id'] == engine_id].sort_values(by='cycle')
        for _, row in engine_data.iterrows():
            yield row.to_dict()
            