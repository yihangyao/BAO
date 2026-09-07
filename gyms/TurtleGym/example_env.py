import os

os.environ["OPENAI_API_KEY"] = "XXX"

if __name__ == "__main__":
    import turtlegym
    config = turtlegym.config.get_default_config()
    config.data_mode = "random" # "single"
    config.data_source = None # "The Disappeared Sister"
    env = turtlegym.env.StoryEnv(config)
    
    for _ in range(17):
        obs, info = env.reset()
    
    print("Observation: ", obs)
    current_story = env.current_story
    print("current story keys:", current_story.keys())
    
    print("="*20)
    print('goal')
    print(current_story['goal'])
    print("="*20)
    print('description')
    print(current_story['description'])
    print("="*20)
    print('ground_truth')
    print(current_story['ground_truth'])
    print("="*20)
    print('evaluation_criteria')
    print(current_story['evaluation_criteria'])
    
    env.close()