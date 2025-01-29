import io
import os
import base64
import json
import yaml
from pathlib import Path
from PIL import Image
from time import time
from src.agents import WebAgent
import numpy as np
from playwright.async_api import Page
from sentence_transformers import SentenceTransformer # For text embedding

import asyncio

# Determine the project root dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define EXAMPLES_DIR relative to the project root
EXAMPLES_DIR = Path(project_root) / "Examples"
EXAMPLE_FILE = "examples.json"

# Combine EXAMPLES_DIR and EXAMPLES_FILE
combined_path = EXAMPLES_DIR / EXAMPLE_FILE

def select_examples(file_path=combined_path, verbose=True):
    ''' Extracts examples from the json formatted list made by the logger '''
    printv = lambda *args: print(*args) if verbose else None
    try:
        with open(file_path, "r") as f:
            examples = [example for example in json.load(f)
                        if (example.get("unannotated_img", None)
                            and example.get("action", None))]
        if not examples:
             printv("WARNING: no good examples found")
        else:
            printv("INFO: Found", len(examples), "examples")
        return examples
    except FileNotFoundError as e:
        print("ERROR: no file found, returning an empty list")
    return []


class FakePage(Page):
    '''
    Emulates a Playwright Page object.
    Since the agent works off a page object, and we don't
    want to load a real webpage in testing, we need a fake
    one with all the methods the agent uses
    '''
    url = ""  # otherwise Page protects it somehow
    def __init__(self, screenshot, url=None):
        self.img = screenshot
        if url is None:
            url = ""
        self.url = url
    
    async def screenshot(self):
        ''' Get the screenshot '''
        return self.img
    
    async def wait_for_load_state(self, *args, **kwargs):
        ''' Pretend to wait for the page to load '''
        return None
    
    async def evaluate(self, java_script):
        ''' Evaluate some javascript on the page, harder to fake '''
        raise NotImplementedError("we don't save the responses for this, and can't fake it yet")
    
def bbox_loc(bbox_set, ind):
    ''' Grab the coordinates of the center of the bbox given by ind '''
    try:
        ind = int(ind)
    except ValueError as e:
        raise TypeError("bbox number is not an integer, it's " + str(ind)
                        + " of type " + str(type(ind)))
    if type(bbox_set) == str:
        bbox_set = yaml.safe_load(bbox_set)
    correct_bbox = bbox_set[ind]
    return (correct_bbox["x"], correct_bbox["y"])


# Initialize sentence tranformer uning all-MiniLM-L6-v2 model
# Model only 80mb so should be fine to initialize directly
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Added helper function for text embedding comparisons
def compute_cosine_similarity(text1 : str, text2 : str):
    '''
    Compute semantic similarity between two pieces of text using sentence embeddings.

    Args:
        text1: First text string 
        text2: Second text string

    Returns:
        float: Cosine similarity  score between 0 and 1, where 1 means identical meaning.
               Measures the distance between two embeddings in vector space (how similar they are)
    '''
    embedding1 = text_model.encode(text1, convert_to_tensor=True)
    embedding2 = text_model.encode(text2, convert_to_tensor=True)

    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return float(similarity)

async def compare_agent_node(
    sample: dict,
    web_agent: WebAgent,
    radius="screen fraction: 0.05",
    semantic_threshold=0.75, # Threshold for cosine similarity to be accepted as a match 
    show_screenshots = False,
    verbose: bool = True):
    ''' 
    Benchmarks the agent node in a WebAgent object
    Args:
        sample: one sample agent run, in the json format the logger saves
        web_agent: a WebAgent object to test
        radius: how to calculate the maximum acceptable error, currently the only option is
                "screen fraction: ###", which is ### fraction of the screen size in each dim
                TODO (Ben): We have the data to check if a click is on the right button, not just in a radius
        show_screenshots: set to True to have the screenshots pop up to compare
        verbose: prints info to visually compare the results
    Returns:
        bool: if the action matched
        float: normalized error, so greater than one is outside the given radius
        float: time taken in seconds
        dict: The full final agent state for analysis
    '''
    # Verify the radius is an acceptable format
    assert radius.split(":")[0].strip() in ["screen fraction"], NotImplementedError("Currently can only handle radii as screen fractions, enter with argument 'screen fraction:0.05' or equivalent")


    # Grab the original image, make a fake page object for it
    clean_img = io.BytesIO(base64.b64decode(sample["unannotated_img"])).read()
    fake_page = FakePage(clean_img)
    sample["page"] = fake_page

    # Grab the original correct answers for the agent
    correct_action = sample["action"]
    correct_args = sample["action_args"]

    # Get location for spatial actions
    correct_loc = None
    if correct_action.lower() in ["click", "type"]:
        correct_loc = bbox_loc(sample["bboxes"], correct_args[0])

    # Run the agent and mark the time taken
    start_time = time()
    end_state = await web_agent.agent.ainvoke(sample)
    end_time = time()

    # Initialize variables for different comparisons
    action_matched = False
    normalized_error = 0
    target_matched = None
    direction_matched = None
    cosine_similarity = None
    text_matched = None

    #Fucntion that compares the location between correct_loc and end_loc
    def compare_action_location(correct_loc, end_loc, radius, clean_img):
        #use nonlocal to access the pre initialized normalized_error find calculate the nomalized error
        nonlocal normalized_error
        if correct_loc:
            fraction = float(radius.split(":")[1].strip())
            screen_x, screen_y = Image.open(io.BytesIO(clean_img)).size
            rad_x = fraction * screen_x
            rad_y = fraction * screen_y
            normalized_error = (((end_loc[0] - correct_loc[0]) / rad_x) ** 2 
                                + ((end_loc[1] - correct_loc[1]) / rad_y) ** 2)    
    
    # Compare actions based on action type
    if end_state["prediction"]["action"].lower() == correct_action.lower():
        action_matched = True
        if correct_action.lower() in ["click", "type"] and correct_loc:
            # Get the end location for click or type actions
            end_loc = bbox_loc(end_state["bboxes"], end_state["prediction"]["args"][0])
            compare_action_location(correct_loc, end_loc, radius, clean_img)

            # Handling when action is type. In this scenario, compute cosine similarity between predicted and actual
            if correct_action.lower() == "type":
                # Get text values
                correct_text = sample["action_args"][1] if len(sample["action_args"]) > 1 else ""
                predicted_text = end_state["prediction"]["args"][1] if len(end_state["prediction"]["args"]) > 1 else ""

                # Compare cosine similarity and check if above threshold
                cosine_similarity = compute_cosine_similarity(correct_text, predicted_text)
                text_matched = cosine_similarity >= semantic_threshold # text_match = True if cosine similarity above threshold

                # Action matches if text matches as well
                action_matched = action_matched and text_matched

        elif correct_action.lower() == "scroll":
            # Extract the correct target and scroll direction
            correct_target = correct_args[0] if len(correct_args) > 0 else ""
            correct_direction = correct_args[1] if len(correct_args) > 1 else ""

            # Ensure "prediction" exists and the action is "scroll"
            if end_state.get("prediction") and isinstance(end_state["prediction"], dict):
            # Extract predicted target and direction only for "scroll" action. If there are no arguments, put an empty string.
                predicted_target = end_state["prediction"].get("args", [])[0] if len(end_state["prediction"].get("args", [])) > 0 else ""
                predicted_direction = end_state["prediction"].get("args", [])[1] if len(end_state["prediction"].get("args", [])) > 1 else ""

            # Compare the predicted and correct targets
            target_matched = correct_target.lower() == predicted_target.lower()

            # Compare the predicted and correct directions
            direction_matched = correct_direction.lower() == predicted_direction.lower()
            
            # Match action if both target and direction match
            action_matched = target_matched and direction_matched

    # Initialize the dictionary to store all results for each run
    results = {
        "action_matched": action_matched,
        "correct_action": correct_action,
        "predicted_action": end_state['prediction']['action'],
        "normalized_error": normalized_error,
        "time_taken": end_time - start_time,
        "target_window_matched": target_matched,
        "direction_matched": direction_matched,
        "cosine_similarity": cosine_similarity,
        "text_matched": text_matched
    }

    # print out a comparison between expected and found actions. Don't be surprised
    # if the box numbers don't match
    if verbose:
        print(f"Time taken: {end_time - start_time:.2f}")
        print("Given:\tPredicted:")
        print(f"{correct_action}\t{end_state['prediction']['action']}")
        if action_matched:    
            if correct_action.lower() == "click" and correct_loc:
                end_loc = bbox_loc(end_state["bboxes"], end_state["prediction"]["args"][0])
                print(f"x: {correct_loc[0]:.0f}\t{end_loc[0]:.0f}")
                print(f"y: {correct_loc[1]:.0f}\t{end_loc[1]:.0f}")
            elif correct_action.lower() == "type" and correct_loc:
                end_loc = bbox_loc(end_state["bboxes"], end_state["prediction"]["args"][0])
                correct_text = correct_args[1] if len(correct_args) > 1 else ""
                predicted_text = end_state["prediction"]["args"][1] if len(end_state["prediction"]["args"]) > 1 else ""
                print(f"Correct Text: {correct_text}\tCorrect Box {correct_loc}")
                print(f"Predicted Text: {predicted_text}\tPredicted Box {end_loc}")
                print(f"Text Similarity: {cosine_similarity:.3f}")
            elif correct_action.lower() == "scroll":
                # TODO: Implement scroll validation
                print(f"Expected scroll: target={correct_target}, direction={correct_direction}")
                print(f"Predicted scroll: target={predicted_target}, direction={predicted_direction}")

    # display annotated images to compare
    if show_screenshots:
        def see_picture(pic):
            return Image.open(io.BytesIO(base64.b64decode(pic)))
        see_picture(sample["screenshot"]).show("original annotation")
        see_picture(end_state["img"]).show("new annotation")

    return results
   
if __name__ == "__main__":
    # Grab some examples
    examples = select_examples()

    # Set up an agent to test
    from src.omniparser import OmniParserConfig, OmniParser
    omniparser = OmniParser.from_config(OmniParserConfig(caption_model="llava"))  # if you want to use llava, set OmniParserConfig to have caption_model="llava"
    # We add tag_with_js=False to skip over the javascript annotation step, since
    # that is currently not implemented in FakePage.evaluate. TODO (Ben): make a FakePage.evaluate
    # method, maybe it could be done by hunting through bboxes for the original labels?
    web_agent = WebAgent(image_parser=omniparser, tag_with_js=False) 

    #Initialize the dictionary to capture the results from each exmaple 
    all_results = []

    # Test them
    for example in examples:
        results = asyncio.run(compare_agent_node(example, web_agent))
        all_results.append(results)
        # Define the output/json file path
    output_file = Path("tests/results.json")

    # Ensure the parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the `all_results` to the JSON file
    try:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Results successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
