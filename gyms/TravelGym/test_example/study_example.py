import travelgym
from travelgym import TravelEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = TravelEnv(config)

# Reset to get initial travel scenario
observation, info = env.reset()
dim = env.current_task.get("dimensions", [])
print(dim)
env.close()