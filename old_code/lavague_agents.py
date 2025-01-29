from src.omniparser import OmniParser

# LaVague imports
from lavague.drivers.selenium import SeleniumDriver
from lavague.core import ActionEngine, WorldModel

# LaVague imports for custom WebAgent class
from lavague.core.agents import WebAgent
from lavague.core.world_model import WorldModel
from lavague.core.action_engine import ActionEngine
from lavague.core.logger import AgentLogger
from lavague.core.token_counter import TokenCounter
from lavague.core.base_engine import ActionResult
from lavague.core.utilities.format_utils import (
    extract_before_next_engine,
    extract_next_engine,
    extract_world_model_instruction,
    replace_hyphens,
)

# Optional imports, if not already covered in the base WebAgent
from typing import Optional, Any
import tempfile
from PIL import Image
import os
import logging

# Set up logging, this is run throughout the WebAgent
logging_print = logging.getLogger(__name__)
logging_print.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format)
logging_print.addHandler(ch)
logging_print.propagate = False

class WebAgentWithOmniParser(WebAgent):
    """This class is built open LaVague's WebAgent class."""
    # TODO(Dominic): Currently doesn't annotate images when doing run_demo or run_gradio. 
    
    def __init__(
        self, 
        world_model: WorldModel, 
        action_engine: ActionEngine, 
        omniparser: OmniParser,
        token_counter: Optional[TokenCounter] = None,
        n_steps: int = 10,
        clean_screenshot_folder: bool = True,
        logger: AgentLogger = None,
    ):
        super().__init__(
            world_model,
            action_engine,
            token_counter,
            n_steps,
            clean_screenshot_folder,
            logger
        )
        self.omniparer = omniparser
        self.annotated_cache = set() # Cache to monitor which screenshots have been annotated. 

    def parse(self, folder_path):
        """Parse and annotate screenshots in `folder_path` which haven't been processed yet. """
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # If filepath is not in the cache, annotate it.
            if file_path not in self.annotated_cache and file_path.endswith('png'):
                try:
                    if self.omniparer:
                        annotated_image, _ = self.omniparer.parse(file_path)
                        print(f"Annotated the image located at: {file_path}")
                    
                    annotated_image.save(file_path)

                    self.annotated_cache.add(file_path)
                except FileNotFoundError as e:
                    print(f"Error finding the file")
                    raise
    
    def run_step(self, objective: str) -> Optional[ActionResult]:
        obs = self.driver.get_obs()
        current_state, past = self.st_memory.get_state()

        # Parse unannotated screenshots
        screenshots_path = obs["screenshots_path"]
        self.parse(screenshots_path)
        world_model_output = self.world_model.get_instruction(
            objective, current_state, past, obs
        )
        logging_print.info(world_model_output)
        next_engine_name = extract_next_engine(world_model_output)
        instruction = extract_world_model_instruction(world_model_output)

        if next_engine_name == "COMPLETE" or next_engine_name == "SUCCESS":
            self.result.success = True
            self.result.output = instruction
            logging_print.info("Objective reached. Stopping...")
            self.logger.add_log(obs)

            self.process_token_usage()
            self.logger.end_step()
            return self.result

        action_result = self.action_engine.dispatch_instruction(
            next_engine_name, instruction
        )
        if action_result.success:
            self.result.code += action_result.code
            self.result.output = action_result.output
        self.st_memory.update_state(
            instruction,
            next_engine_name,
            action_result.success,
            action_result.output,
        )
        self.logger.add_log(obs)

        self.process_token_usage()
        self.logger.end_step()
                

        