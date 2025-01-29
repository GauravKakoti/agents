import base64
import re
from io import BytesIO
from typing import Callable, Dict, Literal

import langchain
import yaml
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from PIL import Image
from playwright.async_api import async_playwright

from src.custom_types import AgentState, Plan
from src.image_hash import ImageHash
from src.llms import (SHOWUI_ACTIONS, SHOWUI_NAV_FORMAT, SHOWUI_NAV_SYSTEM,
                      SHOWUI_TEMPLATE, ChatShowUI)
from src.logger import StructuredLogger
from src.multishot_generator import MultiShotGenerator
# TODO(dominic): We need a better way of choosing between tools...
# from src.nav_tools import (answer, click, copy, enter, go_back, hover, scroll,
#                            select_text, to_google, type_text, wait)
from src.omniparser import OmniParser
# from src.prompts import WEBNAVIGATOR_HEADER, WEBNAVIGATOR_ORDER
from src.prompts import LAVAGUE_HEADER_NAV_ONLY, REFORMULATOR_PROMPT
from src.scraper import SCRAPER_PROMPT, Scraper
from src.tools import click, go_back, scroll, to_google, type_text, wait
from src.utils import (annotate, draw_point, encode_image, format_descriptions,
                       generate_prompt_template, parse_formatter_string,
                       parse_llm_output)

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


class WebAgent:
    """
    A WebAgent class for autonomous web navigation and information extraction.

    Attributes:
        tools (Dict[str, Callable]): Available actions for web interaction. Default tools include:
            - Click: Click elements on page
            - Type: Enter text into fields
            - Scroll: Scroll the page
            - Wait: Pause execution
            - GoBack: Return to previous page
            - Scrape: Extract information
            - Google: Navigate to Google search
        prompt (ChatPromptTemplate): Template for generating agent actions
        llm (BaseChatModel): Language model for decision making, defaults to llama-3.2-90b-vision-preview
        image_parser (OmniParser): Parser for processing page screenshots and identifying UI elements.
            Falls back to mark_page.js if not provided.
        prompt_kwargs (dict[str, str]): input for prompt_template_generator to make a general prompt. Will
            use any prompt sub-section mentioned in the parameter reference by "order"
        multishot_examples (int): number of examples used by multishot prompt generator
        log_screenshots (bool): toggles wheter to save screenshots in the log
        tag_with_js (bool): toggles whether to add javascript-based types to bbox descriptions

    Methods:
        annotate_image: Processes page screenshots to identify interactive elements
        create_agent: Initializes the agent's core decision-making pipeline
        update_scratchpad: Maintains agent's memory of actions and observations
        initialize_browser: Sets up Playwright browser instance
        select_tool: handles tool selection for non-trivial tool names
        build_graph: puts together the graph for langgraph
        compile_graph: compiles the graph put together in build_graph
        run: Executes the agent on a given URL to answer a question
    """

    def __init__(
        self,
        project_root,
        tools: Dict[str, Callable] = None,
        llm=None,
        image_parser: OmniParser = None,
        prompt_kwargs=None,
        multishot_examples=0,
        log_screenshots=True,
        tag_with_js=True,
    ):
        self.project_root = project_root
        # make the prompt, on None builds the lavague-style prompt
        if prompt_kwargs is None or (
            "few_shot" in prompt_kwargs.get("order", ["few_shot"])
            and "few_shot" not in prompt_kwargs
        ):
            # Default multi-shot generation.
            # Making the multishot is currently designed for re-generation each time,
            # but with our current sophistication, this is fine for testing
            few_shot = MultiShotGenerator(project_root=project_root).generate_multishot(
                objective="", url="", num_examples=multishot_examples
            )
            if prompt_kwargs is None:
                prompt_kwargs = {}
            prompt_kwargs["few_shot"] = few_shot
        self.prompt = generate_prompt_template(**prompt_kwargs)

        self.llm = (
            llm if llm is not None else ChatGroq(model="llama-3.2-90b-vision-preview")
        )

        self.structured_logger = StructuredLogger(log_screenshots=log_screenshots)
        self.image_hash = ImageHash()

        self.scraper_tool = Scraper(mm_llm=self.llm, prompt=SCRAPER_PROMPT)
        # Add in the scraper
        if not tools:
            tools = {
                "Click": click,
                "Type": type_text,
                "Scroll": scroll,
                "Wait": wait,
                "GoBack": go_back,
                "Scrape": self.scraper_tool,
                "Google": to_google,
            }
        self.tools = tools

        self.tag_with_js = tag_with_js  # mark whether to attempt to run javascript
        self.image_parser = image_parser
        self.agent = self.create_agent()
        self.graph_builder = StateGraph(AgentState)

        self.build_graph()

        self.graph = self.compile_graph()

    async def annotate_image(self, state):
        result = await annotate(
            state=state,
            parser=self.image_parser,
            use_local_semantics=True,
            tag_with_js=self.tag_with_js,
        )
        return result

    def create_agent(self):
        return (
            self.annotate_image
            | RunnablePassthrough.assign(
                prediction=format_descriptions
                | self.prompt
                | self.llm
                | StrOutputParser()
                | RunnableLambda(
                    lambda agent_output: REFORMULATOR_PROMPT.format(
                        agent_output=agent_output
                    )
                )
                | self.llm
                | StrOutputParser()
                | parse_formatter_string
            )
            | self.structured_logger.log_state
        )

    async def update_scratchpad(self, state: AgentState):
        """
        After a tool is invoked, we want to update
        the scratchpad so the agent is aware of its previous steps.
        We also increment the step counter here.
        """

        state["step"] += 1
        old = state.get("scratchpad")
        if old:
            txt = old[0].content
            step = state.get("step") - 1
        else:
            txt = "Previous action observations:\n"
            step = 1

        page = state["page"]

        # write the observation or append about the failure
        screenshot = await page.screenshot()

        if self.image_hash.imageOutput(screenshot) == False:
            msg = f"\n{step}. [FAILED] {state['observation']}"

            txt += msg

            self.structured_logger.log_tool(
                # observation=state["observation"]+ " - failed",
                observation=msg,
                step=state["step"],
                action=state["prediction"]["action"],
                args=state["prediction"]["args"],
            )

        else:
            txt += f"\n{step}. {state['observation']}"

            self.structured_logger.log_tool(
                # observation=state["observation"]+ " - failed",
                observation=state["observation"],
                step=state["step"],
                action=state["prediction"]["action"],
                args=state["prediction"]["args"],
            )

        return {**state, "scratchpad": [HumanMessage(content=txt)]}

    def select_tool(self, state: AgentState):
        """
        Selects the appropriate tool or node to execute based on the agent's prediction.
        """
        action = state["prediction"]["action"]
        if action.upper() == "ANSWER":
            return END
        if action.upper() == "RETRY":
            return "agent"
        return action

    def build_graph(self):
        self.graph_builder.add_node("agent", self.agent)
        self.graph_builder.add_edge(START, "agent")

        # Add update_scratchpad node
        self.graph_builder.add_node("update_scratchpad", self.update_scratchpad)
        self.graph_builder.add_edge("update_scratchpad", "agent")

        # Add tool nodes and edges
        for node_name, tool in self.tools.items():
            self.graph_builder.add_node(
                node_name,
                # The lambda ensures the function's string output is mapped to the "observation"
                # key in the AgentState
                RunnableLambda(tool)
                | (lambda observation: {"observation": observation}),
            )
            # Always return to the agent (by means of the update-scratchpad node)
            self.graph_builder.add_edge(node_name, "update_scratchpad")

        # Add conditional edges for selecting the next tool
        self.graph_builder.add_conditional_edges("agent", self.select_tool)

    def compile_graph(self):
        return self.graph_builder.compile()

    async def run(
        self,
        start_url: str,
        objective: str,
        max_steps: int = 10,
        headless: bool = False,
    ):
        """Runs the agent on a given web page with a specified input question."""

        final_answer = None

        # Initialize browser
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=headless, args=None)
            page = await browser.new_page()
            await page.set_viewport_size({"width": 1280, "height": 720})

            # Navigate to the start URL and begin the agent loop
            await page.goto(start_url)

            event_stream = self.graph.astream(
                {
                    "step": 1,
                    "page": page,
                    "objective": objective,
                    "scratchpad": [],
                },
                {
                    "recursion_limit": max_steps,
                },
            )

            screenshot = None
            unannotated_img = None
            bboxes = None
            url = start_url
            step_count = -1
            try:
                async for event in event_stream:
                    if "agent" not in event:
                        continue

                    state = event.get("agent")
                    screenshot = state.get("img", None)
                    unannotated_img = state.get("unannotated_img", None)
                    bboxes = yaml.dump(state.get("bboxes", None))
                    step_count = state.get("step", -1)
                    url = state.get("page").url

                    if state["prediction"]["action"].lower() == "answer":
                        final_answer = state["prediction"]["args"][0]

            except Exception as e:
                # Log the error
                self.structured_logger.log_event(
                    objective=objective,
                    url=url,
                    step=step_count,
                    previous_actions="",
                    thoughts="Error",
                    action="Error",
                    action_args=str(e),
                    error=str(e),
                    screenshot=screenshot,
                    bboxes=bboxes,
                    unannotated_img=unannotated_img,
                )
                # save json files before raising
                self.structured_logger.save_to_json()
                self.structured_logger.save_to_json_tools()
                raise

            self.structured_logger.save_to_json()
            self.structured_logger.save_to_json_tools()

        return final_answer if final_answer else "Agent exited inappropriately..."


class WebAgentV2:
    def __init__(
        self,
        orchestrator_prompt,
        formatter_prompt,
        evaluator_prompt,
        mm_llm_model: str = "gpt-4o",
        llm_model: str = "llama-3.3-70b-versatile",
    ):
        # TODO(dominic): A model factory should handle this part
        self.mm_llm = ChatOpenAI(model=mm_llm_model)
        self.llm = ChatGroq(model=llm_model)

        self.orchestrator_agent = orchestrator_prompt | self.mm_llm | StrOutputParser()
        self.formatter = RunnablePassthrough.assign(
            plan=formatter_prompt
            | self.llm
            | StrOutputParser()
            | parse_formatter_string
        )
        self.showui = ChatShowUI()
        self.nav_agent = RunnablePassthrough.assign(
            next_actions=self.get_task
            | SHOWUI_TEMPLATE
            | self.showui
            | StrOutputParser()
            | parse_llm_output
        )
        self.evaluator_agent = evaluator_prompt | self.mm_llm | StrOutputParser()

        self.tools = {
            "CLICK": click,
            "INPUT": type_text,
            "SELECT": click,
            "HOVER": hover,
            "ANSWER": answer,
            "ENTER": enter,
            "SCROLL": scroll,
            "SELECT_TEXT": select_text,
            "COPY": copy,
            # These next three are not known to the agent to be options
            "WAIT": wait,
            "GO_BACK": go_back,
            "GOOGLE": to_google,
        }

        self.graph = self._create_graph()

    def get_task(self, state: AgentState) -> str:
        return {"query": state["plan"]["tasks"][0], "img": state["img"]}

    async def orchestrator_node(
        self, state: AgentState
    ) -> Command[Literal["formatter", "__end__"]]:
        screenshot = await state["page"].screenshot()
        encoded_screenshot = encode_image(screenshot)
        prediction = await self.orchestrator_agent.ainvoke(
            {**state, "img": encoded_screenshot}
        )  # TODO: this is inelegant considering the udpate below...

        # Add logic to determine if we have the answer or need to proceed to formatter
        if "ANSWER" in prediction:
            # TODO(dominic): We probably need to add some logic here to extract answer etc.
            return Command(goto="__end__")

        return Command(
            update={"prediction": prediction, "img": encoded_screenshot},
            goto="formatter",
        )

    async def execute_actions(self, state: AgentState):
        """Function which takes the actions generated by the NavAgent, routes to corresponding functions, and executes"""
        next_actions = state.get("next_actions")
        scratchpad_additions = []

        prev_url = state["page"].url
        exit_code = "SUCCESS"
        page = state["page"]
        try:
            for action in next_actions:
                value, position = action["value"], action["position"]
                # unnormalize position if it is not None
                if position:
                    position = [position[0] * SCREEN_WIDTH, position[1] * SCREEN_HEIGHT]
                observation = await self.tools[action["action"]](page, value, position)
                scratchpad_additions.append(observation)
                # TODO(Ben): screenshot comparator here connected to exit code: STATE_CHANGE_FAILURE
                if prev_url != state["page"].url:
                    exit_code = "URL_CHANGE"
                    break
                if action["action"].lower() == "answer":
                    exit_code = "ANSWER"
                    break
        except Exception as e:
            print(f"Exception: {str(e)}")
            # introduce logic to figure out the exit code
            exit_code = "TOTAL_FAILURE"  # TODO: not all exceptions should necesarily be treated as total_failure, scope this out
            raise

        state["nav_scratchpad"] = scratchpad_additions

        return {**state, "exit_code": exit_code}

    async def evaluator_node(
        self, state: AgentState
    ) -> Command[Literal["orchestrator", "navigator"]]:
        exit_code = state["exit_code"]
        last_task = state["plan"]["tasks"].pop(0)
        txt = state["scratchpad"][0].content if state["scratchpad"] else ""
        task_line = f"\n - Exit code: {exit_code}, Task: {last_task}"
        if (
            exit_code == "TOTAL_FAILURE"
            or exit_code == "URL_CHANGE"
            or (exit_code == "SUCCESS" and not state["plan"]["tasks"])
        ):
            txt += task_line
            return Command(
                update={
                    "scratchpad": [HumanMessage(content=txt)],
                    "nav_scratchpad": [],
                },
                goto="orchestrator",
            )
        elif exit_code == "STATE_CHANGE_FAILURE":
            raise NotImplementedError(
                "Currently STATE_CHANGE_FAILURE shouldn't occur, we need to figure out what to do."
            )
        elif exit_code == "SUCCESS" and state["plan"]["tasks"]:
            updated_screenshot = await state["page"].screenshot()
            updated_encoded_image = encode_image(updated_screenshot)
            last_action = None
            if state["next_actions"]:
                # grab the last action
                idx = len(state["nav_scratchpad"])
                last_action = state["next_actions"][idx - 1]

            if last_action:
                updated_screenshot = draw_point(
                    updated_screenshot, last_action["position"]
                )
            evaluation = await self.evaluator_agent.ainvoke(
                {
                    "previous_tasks": txt,
                    "past_actions": state["nav_scratchpad"],
                    "next_tasks": "\n - ".join(state["plan"]["tasks"]),
                    "objective": state["objective"],
                    "img": encode_image(updated_screenshot),
                }
            )
            # parse evaluation
            thoughts, args = re.split(
                r"\[(?:CONTINUE|REPLAN|MODIFY|ANSWER)\]", evaluation, maxsplit=1
            )  # TODO(dominic): not yet optimized parsing
            case = re.search(r"\[(?:CONTINUE|REPLAN|MODIFY|ANSWER)\]", evaluation)
            if case is not None:
                case = case.group(0)
            else:
                raise Exception

            if case == "[MODIFY]":
                new_plan = state["plan"]
                new_plan["tasks"][
                    0
                ] = args  # the evaluator has determined the next task needs changing
                return Command(
                    update={"plan": new_plan, "img": updated_encoded_image},
                    goto="navigator",
                )
            elif case == "[CONTINUE]":
                return Command(update={"img": updated_encoded_image}, goto="navigator")
            elif case == "[REPLAN]":
                txt += f"\n - Attempted task: {last_task}"
                txt += f"\n - Need to replan because: {args}"
                return Command(
                    update={
                        "scratchpad": [HumanMessage(content=txt)],
                        "nav_scratchpad": [],
                        "img": updated_encoded_image,
                    },
                    goto="orchestrator",
                )
            elif case == "[ANSWER]":
                txt += task_line
                txt += f"\n - [ANSWER]: {args}"
                return Command(
                    update={
                        "scratchpad": [HumanMessage(content=txt)],
                        "img": updated_encoded_image,
                        "nav_scratchpad": [],
                    },
                    goto="orchestrator",
                )
            else:
                NotImplementedError("Evaluator has returned unexpected flag...")

    def _create_graph(self):
        graph_builder = StateGraph(AgentState)

        # Add in the nodes
        graph_builder.add_node("orchestrator", self.orchestrator_node)
        graph_builder.add_edge(START, "orchestrator")

        graph_builder.add_node("formatter", self.formatter)
        # graph_builder.add_edge("orchestrator", "formatter")

        # add in nav agent
        graph_builder.add_node("navigator", self.nav_agent)
        graph_builder.add_edge("formatter", "navigator")

        # add in executor
        graph_builder.add_node("executor", self.execute_actions)
        graph_builder.add_edge("navigator", "executor")

        graph_builder.add_node("evaluator", self.evaluator_node)
        graph_builder.add_edge("executor", "evaluator")

        graph = graph_builder.compile()
        return graph

    async def run(
        self,
        start_url: str,
        objective: str,
        max_steps: int = 10,
        headless: bool = False,
    ):
        # open a browser
        browser = await async_playwright().start()
        # We will set headless=False so we can watch the agent navigate the web.
        browser = await browser.chromium.launch(headless=headless, args=None)
        page = await browser.new_page()
        await page.set_viewport_size({"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT})
        _ = await page.goto(start_url)

        initial_state = {
            "page": page,
            "objective": objective,
            "scratchpad": [],
            "nav_scratchpad": [],
        }

        event_stream = self.graph.astream(initial_state, {"recursion_limit": max_steps})

        async for event in event_stream:
            print(event)

        # TODO(dominic): need to figure out exactly what we want to return
        return


# if __name__ == "__main__":
#     import asyncio

#     from dotenv import load_dotenv

#     load_dotenv()

# from src.omniparser import OmniParserConfig

# omniparser = OmniParser.from_config(OmniParserConfig())

# # Create an instance of WebAgent
# web_agent = WebAgent(
#     # image_parser=omniparser,
#      prompt_kwargs={"header_text": LAVAGUE_HEADER_NAV_ONLY}
# )

# # Define the entry point for running the agent
# async def main():
#     start_url = "https://www.google.com"
#     question = "What is the weather like in London?"
#     max_steps = 20  # Adjust as needed
#     headless = False  # Set to True if you don't need to see the browser

#     # Run the agent and capture the final answer
#     final_answer = await web_agent.run(start_url, question, max_steps, headless)

#     # Print the final answer
#     print("\nFinal Answer:", final_answer)

# # Run the main function in the asyncio event loop
# asyncio.run(main())
