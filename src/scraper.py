import json
import os
from typing import List
import asyncio
import aiofiles

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, Runnable

from src.custom_types import AgentState


class PriceInformation(BaseModel):
    """Information about a pricing"""
    price: str = Field(..., description="The price value as a string (e.g. $19.99)")
    currency: str = Field(..., description="The currency symbol or code (e.g. $ or USD)")
    description: str = Field(..., description="A brief description of the item or service")

class Prices(BaseModel):
    """Identifying all pricing information present in the screenshot or text."""
    prices: List[PriceInformation] = Field(..., description="A list of pricing information found on the page")


parser = PydanticOutputParser(pydantic_object=Prices)

SCRAPER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user", 
            [
                {
                    "type": "text",
                    "text": """You are a specialized AI who is a part of a multi-agent system tasked with the scraping pricing information from screenshots of websites.
                    Below you are given a screenshot of a website. Analyze the image and extract all pricing details.
                    Return them as a JSON-formatted string that conforms to the following schema:
                    {format_instructions}
                    Return only a JSON-formatted string and nothing else.
                    If no pricing information is found, return `None`."""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,{img}"}
                }
            ]
        )
    ]
).partial(format_instructions=parser.get_format_instructions())

class Scraper(Runnable):
    def __init__(self, mm_llm, prompt, save_file: str = "scraped_data.jsonl"):
        super().__init__()
        self.chain = prompt | mm_llm
        self.save_file = save_file

        if not os.path.isfile(self.save_file):
            with open(self.save_file, "w"):
                pass

    async def run(self, state: AgentState):
        img_data = state.get('img')
        if not img_data:
            raise ValueError("Image path is missing from the state.")
        
        
        print(f"Passing image to chain...")
        result = await self.chain.ainvoke({"img": img_data}) ;        
        state['observation'] = result.content # TODO(dominic): is this the best way to do this? Or should we modify how we record observations...

        async with aiofiles.open(self.save_file, "a") as f:
            json_line = json.dumps(result.content)
            await f.write(json_line + "\n")

        return state

    async def ainvoke(self, state: AgentState, *args, **kwargs):
        return await self.run(state)

    def invoke(self, state: AgentState, *args, **kwargs):
        # TODO(dominic) probably not best practice to do this...
        return asyncio.create_task(self.run(state))
    
    async def __call__(self, state: AgentState):
        ''' 
        Make a function of this object to match how tools are designed.
        TODO(ben): this is currently bad, it should interface with the
        rest of the class better
        '''
        img_data = state.get('img')
        if not img_data:
            raise ValueError("Image path is missing from the state.")
        result = await self.chain.ainvoke({"img": img_data})
        obs = result.content
        async with aiofiles.open(self.save_file, "a") as f:
            json_line = json.dumps(result.content)
            await f.write(json_line + "\n")
        # with open("quick-log", "a+") as f:  # Quick debugging file to get some info on the output
        #     f.write("OBSERVATION WAS " + str(obs))
        return "Scraped and found " + str(obs)
