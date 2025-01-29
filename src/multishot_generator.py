from pathlib import Path
import json
import yaml
import numpy as np
from src.utils import format_descriptions
EXAMPLE_FILE = "examples.json"

HEADER_TEXT = "### Examples:\n\n"
ONE_SHOT_TEMPLATE = """Example {num}:
objective: {objective}
previous_actions: {previous_actions}
bboxes: {formatted_bboxes}

Response:
Thoughts: {thoughts}
Action: {action}  {formatted_action_args}
"""

class MultiShotGenerator:
    def __init__(self, project_root,
                 example_files = EXAMPLE_FILE,
                 header_text = HEADER_TEXT,
                 one_shot_template = ONE_SHOT_TEMPLATE):
        
        self.examples_dir = Path(project_root + "/Examples/")
        self.examples = self._load(example_files)
        self.header_text = header_text
        self.one_shot_template = one_shot_template

    def generate_multishot(self, objective, url, num_examples=8, min_bboxes=5, max_bboxes=15):
        '''
        Generate the multishot portion of the prompt 
        POSSIBLE FUTURE ISSUE, if the logger saves anything called "num", 
        "formatted_bboxes", or "formatted_action_args", we will have an error
        '''
        examples = self._select_examples(objective, url, num_examples)
        shots = []
        for i, example in enumerate(examples):
            example["num"] = i
            bboxes, correct_box_index = self._select_bboxes(example["bboxes"], example["action_args"])
            
            formatted_bboxes = format_descriptions({"bboxes": bboxes})["bbox_descriptions"]
            example["formatted_bboxes"] = formatted_bboxes
            example["formatted_action_args"] = self._format_action_args(example["action_args"], correct_box_index)
            shots.append(self.one_shot_template.format(**example))
        return self.header_text + "\n".join([shot for shot in shots])

    def _load(self, file_names):
        ''' read example file(s) '''
        # If it's just one string/path, make it a single-element list for compatibility
        if type(file_names) == str or type(file_names) == Path:
            file_names = [file_names]

        # On each file, load in the data
        examples = []
        for file_name in file_names:
            with open(self.examples_dir / file_name, "r") as f:
                json_data = json.load(f)
                examples += json_data
        return examples

    def _select_examples(self, objective, url, num_examples=8):
        '''
        Select a subset of the examples given the current location.
        Currently this basically does nothing, but we might want to have
        it somewhat intelligently select examples based on similarity of
        objective (eg split objectives into categories or some such),
        the url if we've been here before, and select for a diversity of
        decisions, so it can see when it might want to use a given tool
        '''
        return np.random.choice(self.examples, 
                                min(num_examples, len(self.examples)), 
                                replace=False)

    def _select_bboxes(self, bboxes, action_args, min_samples=5, max_samples=15):
        ''' Select the correct bounding box along with a few others, somewhere
        between min_samples and max_samples, hard limited at the number of boxes'''

        de_yamlfy_bboxes = yaml.safe_load(bboxes)

        # How many bboxes do we want
        num_samples = np.random.randint(min_samples, max(min_samples, min(len(de_yamlfy_bboxes), max_samples)))
        # We MUST have the correct answer
        correct = de_yamlfy_bboxes.pop(action_args[0])
        correct["is correct"] = True

        # Pick the examples other than the correct choice
        samples = np.random.choice(de_yamlfy_bboxes,
                                min(num_samples, len(de_yamlfy_bboxes)),
                                replace=False)
        
        # Combine the random boxes and the correct one and scramble it
        bboxes = np.concatenate([samples, [correct]], axis=0)
        np.random.shuffle(bboxes)

        correct_index = -1
        for i, entry in enumerate(bboxes):
            if "is correct" in entry:
                correct_index = i
                break
            
        return bboxes, correct_index
    
    def _format_action_args(self, action_args, new_box_num=None):
        '''
        Format the action args ready for the prompt,
        including adjusting the box number (arg[0]) to be new_box_num
        '''
        if new_box_num is None:
            new_box_num = action_args[0]
        elif type(new_box_num) != str:
            new_box_num = str(new_box_num)
        return "; ".join([new_box_num] + action_args[1:])