from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any, Tuple, List
from ast import literal_eval
from copy import deepcopy
import json
import time
import asyncio
import random
import re

# ANSWER_SYS = """## **Task**
# You are a helpful agent to help me evaluate the correctness of the user's story against the ground truth in a Turtle Soup game. You should give both your score and evaluation feedback based on a evaluation protocol provided. Please follow the instructions below.

# ## **Instructions**
# 1. There may exist multiple evaluation criteria based on the evaluation protocol. You should give a score for each criteria.
# 2. Your score can only take two values: 0, 1.0, where 0 means the user's answer is incorrect, or does not explicitly points to any answer, and 1.0 means the user's answer is completely correct and explicitly aligns with ground-truth with explicit evidance. If you cannot provide an exact quote from the user answer that matches the criterion, the score must be 0.
# 3. After giving the score, you should give an overall feedback about which part in the user's answer is correct (or the story is all wrong and totally not aligned). Do not say whcih part is incorrect ot not aligned with the ground truth. Do not release anything else about the ground truth (bottom) or the evaluation protocol. Try to keep your feedback concise and to the point. Be conservative when concluding that an answer covers the ground truth. Assign a non-negative result only if the answer explicitly and unambiguously states the ground truth. 

# ## **Example Format**

# ### Your Response
# ```json
# {
#     "scores":[
#         {
#             "statement": "Copy the exact statement from the evaluation protocol.",
#             "thought": "Your thought about how to evaluate the statement, and justify the score you will give based on the protocol statement and comparison between the ground truth and user's answer. Align them with explicit evidence if you think a part of the response is correct.",
#             "score": 0 or 1.0
#         },
#         ... (the number of scores should be the same as the number of criteria in the evaluation protocol, and the order should also exactly match)
#     ],
#     "feedback": "Your feedback to the user's answer about which part is correct. Do not release anything about the ground truth (bottom) and the evaluation protocol. Be concise and to the point. Use the second person tone (you / your) to address the user."
# }
# ```
# """

ANSWER_SYS = """## **Task**
You are a helpful agent to help me evaluate the correctness of the user's story against the ground truth in a Turtle Soup game. You should give both your score and evaluation feedback based on a evaluation protocol provided. Please follow the instructions below.

## **Instructions**
1. There may exist multiple evaluation criteria based on the evaluation protocol. You should give a score for each criteria.
2. Your score can only take three values: 0, 0.5, 1.0, where 0 means the user's answer is completely incorrect (not even close to the ground truth), 0.5 means the user's answer partially aligns with the ground truth, and 1.0 means the user's answer is completely correct.
3. After giving the score, you should give an overall feedback about which part in the user's answer is correct (or the story is all wrong and totally not aligned). Do not say whcih part is incorrect ot not aligned with the ground truth. Do not release anything else about the ground truth (bottom) or the evaluation protocol. Try to keep your feedback concise and to the point.

## **Example Format**

### Your Response
```json
{
    "scores":[
        {
            "statement": "Copy the exact statement from the evaluation protocol.",
            "thought": "Your thought about how to evaluate the statement, and justify the score you will give based on the protocol statement and comparison between the ground truth and user's answer.",
            "score": 0 or 0.5 or 1.0
        },
        ... (the number of scores should be the same as the number of criteria in the evaluation protocol, and the order should also exactly match)
    ],
    "feedback": "Your feedback to the user's answer about which part is correct. Do not release anything about the ground truth (bottom) and the evaluation protocol. Be concise and to the point. Use the second person tone (you / your) to address the user."
}
```
"""

ANSWER_USER = """## **Note**
- Please return in JSON format wrapped in ```json ... ```. Make sure your json content could be parsed and contains the field required as instructed in the example format.
- The number of elements in the "scores" list should be the same as the number of criteria in the evaluation protocol, and the order of the scores should be the same as the order of the criteria in the evaluation protocol, with the statement texts exactly matched.
- You should strictly verify whether the user's answer explicitly states each required criterion in the evaluation protocol by checking for exact textual evidence in the user's answer. Do not infer intent, do not fill in missing details, and do not rely on plausibility or world knowledge. If explicit evidence is missing, the score must be 0 for this term.
- In your feed back, please be concise and to the point only about which part is correct. Do not release anything about the ground truth (bottom) or the evaluation protocol. Use the second person tone (you / your) to address the user.

### Turtle Soup Story (Surface)
<surface>

### Ground Truth (Bottom)
<bottom>

### User's Answer Story (Bottom)
<user>

### Evaluation Protocol
<protocol>

### Your Response
"""

ACTION_SYS = """## **Task**
You are a helpful assistant to respond to the user query based on the given story scenario (surface) and ground truth (bottom) in a Turtle Soup game. Please follow the instructions below.

## **Instructions**
1. You can only give three values: "Yes", "No", or "Maybe" in your response.
2. "Yes" means the user's query or stated scenario is completely correct according (or aligned) to the ground truth (bottom) of the stroy.
3. "No" means the user's query is incorrect or contradicts the ground truth (bottom) of the story, or the user's query is not even close to the ground truth.
4. "Maybe" means the user's query is can be correct or incorrect, it is hard to tell and not clearly stated in both the bottom and the surface of this story. "Maybe" is usually used when the user's query is not quite relevant to the ground truth. Please try to be determinant and use as less "Maybe" in your response as possible.

## **Example Format**

### Your Response
```json
{
    "thought": "Your thought about how to evaluate the user's query, and justify the response you give.",
    "response": "Yes" or "No" or "Maybe"
}
```
"""

ACTION_USER = """## **Note**
- Please return in JSON format wrapped in ```json ... ```. Make sure your json content could be parsed and contains the field required as instructed in the example format.
- You should carefully reason about the user's query and compare it with the ground truth based on the story scenario before giving the response.
- Please try to be fair and objective in your evaluation. Please use less "Maybe" in your response as possible unless it is hard to tell or indeed not relevant to the ground truth of the story.

### Turtle Soup Story (Surface)
<surface>

### Ground Truth (Bottom)
<bottom>

### User's Query
<query>

### Your Response
"""

import json
import os

def save_messages_for_debug(messages, debug_path):
    # Create parent directory if needed
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)

    # Only save if file does NOT exist
    if not os.path.exists(debug_path) or True:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        print(f"[Debug] Saved messages to {debug_path}")
    else:
        print(f"[Debug] File already exists, not overwriting: {debug_path}")


def build_prompt(action_str: str, is_answer: bool = False, story: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Build the prompt for the LLM to evaluate agent actions."""
    if is_answer:
        # For answers, use the evaluation criteria to score the response
        criteria = deepcopy(story['evaluation_criteria'])
        for cr in criteria:
            cr.pop("score", None) # we do not want the weight to be confused with the score
        criteria_json = json.dumps(criteria, indent=4)

        system_prompt = ANSWER_SYS
        user_prompt = ANSWER_USER.replace("<protocol>", criteria_json).replace("<surface>", story['description']).replace("<bottom>", story['ground_truth']).replace("<user>", action_str[8:].strip())
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # save_messages_for_debug(messages=messages, debug_path="/proj/inf-scaling/yihang/personalization/UserRL/debug_answer.json") 
        return messages
    else:
        # For actions, evaluate if they are helpful for understanding the story
        system_prompt = ACTION_SYS
        user_prompt = ACTION_USER.replace("<surface>", story['description']).replace("<bottom>", story['ground_truth']).replace("<query>", action_str[8:].strip())
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # save_messages_for_debug(messages=messages, debug_path="/proj/inf-scaling/yihang/personalization/UserRL/debug_non_answer.json")
        return messages



def extract_json(response_text):
    # Try extract fenced JSON
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, flags=re.DOTALL)
    if match:
        return match.group(1)

    # Fallback: extract the first JSON object
    match = re.search(r"(\{[\s\S]*\})", response_text)
    if match:
        return match.group(1)

    raise ValueError("No JSON found in response")

def parse_llm_response(response_text: str, is_answer: bool = False, story: Dict[str, Any] = None) -> Tuple[str, float, Dict[str, Any]]:
    """Parse LLM response to extract feedback, reasoning, score, and satisfied criteria."""
    # if response_text.startswith("```json"): # TODO format here
    #     response_text = response_text[7:]
    # if response_text.endswith("```"):
    #     response_text = response_text[:-3]
    # response_json = literal_eval(response_text)
    
    try:
        json_block = extract_json(response_text)
        response_json = json.loads(json_block)
    except Exception as e:
        print("[Parse Error] Raw output:", repr(response_text))
        raise e

    if is_answer:
        evaluation_criteria = story['evaluation_criteria']
        scores = response_json["scores"]
        assert len(scores) == len(evaluation_criteria), "The number of scores should be the same as the number of criteria in the evaluation protocol"

        overall_score = 0.0
        feedback = ""
        for protocol, score in zip(evaluation_criteria, scores):
            weight = protocol["score"]
            judgment = score["score"]
            overall_score += weight * judgment
            if judgment == 1: # how about 0.5? The original repo ignores 0.5
                statement = protocol["statement"]
                feedback += f"{statement}: Covered\n"
            # TODO if judgement == 0.5
        
        feedback = feedback.strip()
        if feedback == "":
            feedback = "Your answer does not effectively cover any evaluation points."
        else:
            feedback = "Here's the evaluation points that your story effectively covers:\n" + feedback + "\nThere might be more hidden twists that you haven't covered. Try to continue ask related questions and answer with creativity."

        # feedback = response_json["feedback"]
        return feedback, overall_score, response_json
    else:
        # Parse answer evaluation response
        feedback = response_json["response"]
        return feedback, 0.0, response_json
    

def evaluate_action(action_str: str, story: Dict[str, Any], model_config: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any], bool]:
    """
    Evaluate an agent action using the LLM.
    
    Returns:
        Tuple of (feedback, reasoning, success, score, satisfied_criteria)
        - feedback: "yes", "no", "excellent", "good", "partial", "poor", "incorrect"
        - reasoning: LLM's explanation
        - success: whether the API call was successful
        - score: numerical score (0-1) for answers, 0 for actions
    """

    if action_str.startswith("[finish]"):
        return "", 0.0, None, True

    is_answer = action_str.startswith("[answer]")
    
    try:
        # Create OpenAI client
        # it was a url list
        openai_user = os.environ.get("USE_OEPNAI_USER", None)
        openai_user_key =  os.environ.get("OPENAI_API_KEY", None)
        if openai_user and openai_user_key and "gpt" in model_config["model_name"]:
            model_name = model_config["model_name"]
            print(f"using Open AI model {model_name} as user model!")
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1")
            
        else:
        
            urls = os.environ.get("URL_LIST", None)
            if urls:
                # self-hosted models on other machines.
                urls = urls.split(",")
                base_url_list = [f"http://{u}:8000/v1" for u in urls]
            
            else:
                # self-hosted models on the same machines
                ports = [
                    "30001",
                    "30002",
                    "30003",
                    "30004",
                ]
                base_url_list = [f"http://127.0.0.1:{p}/v1" for p in ports]
                
            base_url = random.choice(base_url_list)
            model_name = model_config["model_name"]
            time_out = model_config["timeout"]
            
            print("base_url:", base_url)
            print("model name:", model_name)
            print("time_out:", time_out)
            print("="*20)
            
            client = OpenAI(api_key=model_config["api_key"], base_url=base_url)
        
        # Build the prompt
        messages = build_prompt(action_str, is_answer, story)
        
        try_time = 0
        while try_time < 2:
            try:
                # Make API call using the new client format
                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    temperature=model_config["temperature"],
                    max_tokens=model_config["max_tokens"],
                    timeout=model_config["timeout"]
                )
                break
            except Exception as e:
                print(f"inside Error evaluating action: {e}")
                try_time += 1
                time.sleep(2)
                if try_time >= 3:
                    raise e
        
        # Extract response text
        response_text = response.choices[0].message.content.strip()

        # Parse the response
        feedback, score, json_response = parse_llm_response(response_text, is_answer, story)
        
        return feedback, score, json_response, True
        
    except Exception as e:
        print(f"[TurtleGym] outside Error evaluating action: {e}")
        return "Feedback not available currently. Please try again.", 0.0, None, False


async def evaluate_action_async(action_str: str, story: Dict[str, Any], model_config: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any], bool]:
    """
    Evaluate an agent action using the LLM (async version).
    
    Returns:
        Tuple of (feedback, reasoning, success, score, satisfied_criteria)
        - feedback: "yes", "no", "excellent", "good", "partial", "poor", "incorrect"
        - reasoning: LLM's explanation
        - success: whether the API call was successful
        - score: numerical score (0-1) for answers, 0 for actions
    """

    if action_str.startswith("[finish]"):
        return "", 0.0, None, True

    is_answer = action_str.startswith("[answer]")
    
    try:
        openai_user = os.environ.get("USE_OEPNAI_USER", None)
        openai_user_key =  os.environ.get("OPENAI_API_KEY", None)
        if openai_user and openai_user_key and "gpt" in model_config["model_name"]:
            model_name = model_config["model_name"]
            print(f"using Open AI model {model_name} as user model!")
            client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1")
            
        else:
        
            urls = os.environ.get("URL_LIST", None)
            if urls:
                # self-hosted models on other machines.
                urls = urls.split(",")
                base_url_list = [f"http://{u}:8000/v1" for u in urls]
            
            else:
                # self-hosted models on the same machines
                ports = [
                    "30001",
                    "30002",
                    "30003",
                    "30004",
                ]
                base_url_list = [f"http://127.0.0.1:{p}/v1" for p in ports]
            
            base_url = random.choice(base_url_list)
            model_name = model_config["model_name"]
            time_out = model_config["timeout"]
            
            print("base_url:", base_url)
            print("model name:", model_name)
            print("time_out:", time_out)
            print("="*20)
            
            client = AsyncOpenAI(api_key=model_config["api_key"], base_url=base_url)
        
        # Build the prompt
        messages = build_prompt(action_str, is_answer, story)
        
        try_time = 0
        while try_time < 2:
            try:
                # Make async API call
                response = await client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    temperature=model_config["temperature"],
                    max_tokens=model_config["max_tokens"],
                    timeout=model_config["timeout"]
                )
                break
            except Exception as e:
                print(f"inside Error evaluating action (async): {e}")
                try_time += 1
                await asyncio.sleep(2)  # Use async sleep
                if try_time >= 3:
                    raise e
        
        # Extract response text
        response_text = response.choices[0].message.content

        # Parse the response
        feedback, score, json_response = parse_llm_response(response_text.strip(), is_answer, story) # TODO: Debug here
        
        return feedback, score, json_response, True
        
    except Exception as e:
        print(f"[TurtleGym] outside Error evaluating action (async): {e}")
        # print(f"[TurtleGym debug] message: {messages}")
        print(f"[TurtleGym debug] response_text: {response_text}")
        return "Feedback not available currently. Please try again.", 0.0, None, False

