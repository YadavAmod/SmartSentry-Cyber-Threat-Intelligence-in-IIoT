# utils/data_simulator.py
import numpy as np
import pandas as pd

class IoTDataSimulator:
    def __init__(self):
        self.normal_params = {
            'packet_size': (500, 100),  # mean, std
            'protocol': ['TCP', 'UDP', 'ICMP'],
            'port': list(range(1, 1025))
        }
        
    def generate_data(self, n_samples=1):
        """Generate simulated IoT device data"""
        data = {
            'packet_size': np.random.normal(
                self.normal_params['packet_size'][0],
                self.normal_params['packet_size'][1],
                n_samples
            ),
            'protocol': np.random.choice(
                self.normal_params['protocol'],
                n_samples
            ),
            'port': np.random.choice(
                self.normal_params['port'],
                n_samples
            )
        }
        return pd.DataFrame(data)