from src.utils import parse

def test_parser_action_finding() -> None:
    ''' Test that the parser can reliably find actions '''
    parsed_dict = parse("""**Thoughts:** I have found the holy grail in box 1
                        
                         **Action:** - Click 1
                        """)
    assert parsed_dict["args"] is not None
    assert parsed_dict["action"] == "Click" 
    assert parsed_dict["args"][0] == "1"
    parsed_dict = parse("""thought:I am at the data location, so we need to scrape
action: scrape""")
    assert parsed_dict["action"] == "scrape"
    assert parsed_dict["args"] is None

def test_parser_arg_splitting() -> None:
    ''' Make sure the parser can split actions correctly '''
    parsed_dict = parse("""**Thoughts:** Next enter San francisco in the search bar to check the weather
                        
                         **Action:**- Type 1,San Francisco
                        """)
    assert parsed_dict["args"] is not None
    assert parsed_dict["action"] == "Type" 
    assert parsed_dict["args"][0] == "1"
    assert parsed_dict["args"][1] == "San Francisco"
    parsed_dict = parse("""**Thoughts:** Next enter San francisco in the search bar to check the weather
                        
                         **Action:** Type: 1 "San Francisco"
                        """)
    assert parsed_dict["args"][0] == "1"
    assert parsed_dict["args"][1] == "San Francisco"
    parsed_dict = parse("""**Thoughts:** We need to give the bank some information about us, so enter 800 in the credit score box
                        
                         **Action:** Type 17;       '"800"'
                        """)
    assert parsed_dict["args"][0] == "17"
    assert parsed_dict["args"][1] == "800"
    parsed_dict = parse("""**Thoughts:** We need to give the bank some information about us, so enter 800 in the credit score box
                        
                         **Action:** Type 17 800    
                        """)
    assert parsed_dict["args"][0] == "17"
    assert parsed_dict["args"][1] == "800"


def test_parser_thought_finding() -> None:
    ''' Test that the parser can reliably find thoughts '''
    parsed_dict = parse("""
                        **Thoughts:** I thought the holy grail was in box 1
                        
                         **Action:** Click 1""")
    assert parsed_dict["thought"]=="I thought the holy grail was in box 1"
    parsed_dict = parse("""thought I think we need to search "San Francisco"
                        
                         action Click 1""")
    assert parsed_dict["thought"]=='I think we need to search "San Francisco"'


def test_parser_edge_cases() -> None:
    ''' test that the parser handles edge cases appropriately '''
    parsed_dict = parse("""
                        I'm not going to engage in this topic matter
                        """)
    print(parsed_dict["args"])
    assert parsed_dict["thought"] == "I'm not going to engage in this topic matter"
    assert parsed_dict["action"] == "retry"
    assert parsed_dict["args"] == "Could not parse LLM Action in: I'm not going to engage in this topic matter,\nWe will mark this as the thought"