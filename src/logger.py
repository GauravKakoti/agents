import json
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Log file paths
LOGS_DIR = Path("Logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)



# Structured Logger Class
class StructuredLogger:
    def __init__(
        self,
        log_file_path=LOGS_DIR / "logs.log",
        log_tool_path = LOGS_DIR /"tool_logs.json",
        structured_logs_json_path=LOGS_DIR / "structured_logs.json",
        log_screenshots=True,
        suppress_api_info=True
    ):
        # Initialize paths and directories
        self.log_file_path = log_file_path
        self.structured_logs_json_path = structured_logs_json_path
        self.log_tool_path = log_tool_path
        # Flag for if we want to save screenshots along the way
        self.log_screenshots = log_screenshots

        # Initialize log entries
        self.log_entries = []
        self.logs = []

        # Configure basic logging for console and file output
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
            handlers=[logging.FileHandler(self.log_file_path), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("WebAgentLogger")

        # Suppress all the api call logging
        
        if suppress_api_info:
            logging.getLogger('httpx').setLevel(logging.WARNING)
            logging.getLogger('httpcore').setLevel(logging.WARNING)

    def log_event(
        self,
        objective: str,
        url: str,
        step: int,
        previous_actions: str,
        thoughts: str,
        action: str,
        action_args: str,
        error: str = None,
        screenshot: str = None,
        bboxes: str = None,
        unannotated_img: str = None
    ):
        """Logs events with detailed information, including optional screenshots."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "objective": objective,
            "url": url,
            "step": step,
            "previous_actions": previous_actions,
            "thoughts": thoughts,
            "action": action,
            "action_args": action_args,
            "error": error,
            "screenshot": screenshot if self.log_screenshots else None,
            "bboxes": bboxes,
            "unannotated_img": unannotated_img
        }

        # Append log entry to internal list for structured logging
        self.log_entries.append(log_entry)

        # Log to console/file
        self.logger.info(f"Step {step} | Action: {action} | Action Args: {action_args}")
        if error:
            self.logger.error(f"Error: {error}")

    def save_to_json(self, path=None):
        """Saves structured log entries to a JSON file for additional structure."""
        if path is None:
            path = self.structured_logs_json_path
        with open(path, "w") as f:
            json.dump(self.log_entries, f, indent=4)
        self.logger.info(
            f"Structured logs saved to JSON at {path}"
        )
        
    def save_to_json_tools(self, path=None):
        """Saves structured log tool entries to a JSON file."""
        if path is None:
            path = self.log_tool_path
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=4)        
        self.logger.info(
            f"Structured log tools saved to JSON at {path}"
        )

    def log_api_response(self, response):
        """Logs API responses."""
        self.logger.info(f"API Response: {response}")

    def log_state(self, state):
        """
        Interface to intercept the state function and log it
        """
        pred = state.get("prediction", {})
        scratchpad = state.get("scratchpad", None)
        prev_actions = scratchpad[0].content if scratchpad else ""

        # Verifica si pred es un diccionario antes de acceder a sus claves
        if isinstance(pred, dict):
            thoughts = pred.get("thoughts", None)
            action = pred.get("action", None)
            args = pred.get("args", None)
        else:
            self.logger.debug(f"Unexpected prediction type: {type(pred)}, value: {pred}")
            thoughts = None
            action = None
            args = None

        self.log_event(
            state.get("objective", ""),
            state.get("page").url if "page" in state else "",
            state.get("step", -1),
            prev_actions,
            thoughts,
            action,
            args,
            screenshot=state["img"] if "img" in state else None,
            bboxes=yaml.dump(state["bboxes"]) if "bboxes" in state else None,
            unannotated_img=state["unannotated_img"] if "unannotated_img" in state else None
        )
        return state
    
    def log_tool(self, 
                observation: str,
                step: int,
                action: str,
                args: str):
        ''' log action from a tool in the agent '''
        
        logInfo = {
            "timestamp": datetime.now().isoformat(),
            "observation": observation,
            "step": step,
            "action": action,
            "args": args
            }
        
        self.logs.append(logInfo)

        
    def clear_log(self):
        ''' Empty the log '''
        self.log_entries = []