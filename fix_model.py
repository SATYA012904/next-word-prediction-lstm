import h5py
import json

filename = 'lstm_model.h5'

def clean_config(config):
    """Recursively remove Keras 3 specific keys that break Keras 2."""
    if isinstance(config, dict):
        # 1. Swap batch_shape for Keras 2's batch_input_shape
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        
        # 2. List of keys to delete entirely
        keys_to_delete = ['optional', 'quantization_config', 'registered_name', 'module']
        for key in keys_to_delete:
            config.pop(key, None)
        
        # 3. Handle the DTypePolicy object (Keras 3) vs plain string (Keras 2)
        if 'dtype' in config and isinstance(config['dtype'], dict):
            # Extract just the name (e.g., 'float32') and discard the rest
            config['dtype'] = config['dtype'].get('config', {}).get('name', 'float32')

        # Run recursively for nested dictionaries and lists
        for key, value in config.items():
            clean_config(value)
    elif isinstance(config, list):
        for item in config:
            clean_config(item)
    return config

try:
    with h5py.File(filename, 'r+') as f:
        if 'model_config' in f.attrs:
            config_data = f.attrs['model_config']
            if isinstance(config_data, bytes):
                config_data = config_data.decode('utf-8')
            
            config_dict = json.loads(config_data)
            
            # Clean the entire configuration tree
            fixed_config = clean_config(config_dict)
            
            f.attrs['model_config'] = json.dumps(fixed_config).encode('utf-8')
            print("✅ Model deep-cleaned! Keras 3 specific arguments removed.")
        else:
            print("❌ Could not find model_config.")
except Exception as e:
    print(f"❌ Error: {e}")