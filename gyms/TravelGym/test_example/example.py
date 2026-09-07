import travelgym
from travelgym import TravelEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = TravelEnv(config)

# Reset to get initial travel scenario
observation, info = env.reset()
print(f"User: {observation['feedback']}")

# Ask about travel preferences
obs, reward, terminated, truncated, info = env.step("[action] What destination are you traveling to?")
print(f"User: {obs['feedback']}")
print(f"Reward: {reward}")

# Search for hotel options
obs, reward, terminated, truncated, info = env.step('[search] {"function_name": "hotel", "arguments": {"location": "Los Angeles", "room_type": "Queen"}}')
print(f"Search results: {obs['feedback']}")

# Make a recommendation
obs, reward, terminated, truncated, info = env.step("[answer] H13")
print(f"Result: {obs['feedback']}")
print(f"Final reward: {reward}")

env.close()