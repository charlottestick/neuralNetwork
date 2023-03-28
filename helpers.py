from tensorflow import config

def reportDevice() -> None:
    computeDevice = 'GPU' if len(config.list_physical_devices('GPU')) > 0 else 'CPU'
    print('\nUsing', computeDevice, '\n')