from langchain.tools import BaseTool
from codeinterpreterapi import CodeInterpreterSession



class CodeInterpreterTool(BaseTool):
    name = "Coding-Sandbox"
    description = """
    Tool is a Code Interpreter powered by GPT-4, designed to assist with a wide range of tasks, particularly those related to data science, data analysis, data visualization, and file manipulation.

    Tool can perform a variety of other tasks. Here are some examples:

    - Project Management: Tool can assist mapping out project steps.
    - Mathematical Computation: Tool can solve complex math equations.
    - Document Analysis: Tool can analyze, summarize, or extract information from large documents.
    - Code Analysis and Creation: Tool can analyze and critique code, and even create code from scratch.
    - Many other things that can be accomplished running python code in a jupyter environment.

    Tool can execute Python code within a sandboxed Jupyter kernel environment. Tool comes equipped with a variety of pre-installed Python packages including numpy, 
    pandas, matplotlib, seaborn, scikit-learn, yfinance, scipy, statsmodels, sympy, bokeh, plotly, dash, and networkx. Additionally, Tool has the ability to use other packages which automatically get installed when found in the code.

    Please note that Tool is designed to assist with specific tasks and may not function as expected if used incorrectly. If you encounter an error, please review your code and try again. 
    After two unsuccessful attempts, Tool will simply output that there was an error with the prompt.

    Remember, Tool is constantly learning and improving. Tool is capable of generating human-like text based on the input it receives, engaging in natural-sounding conversations, 
    and providing responses that are coherent and relevant to the topic at hand. Enjoy your coding session!
    """

    def _run(self, tool_input):
        code = tool_input  # Assuming the tool_input is the code to be run
        return CodeInterpreterAPIWrapper()._run_code(code)

class CodeInterpreterAPIWrapper:
    def _run_code(self, code):
        with CodeInterpreterSession() as session:
            response = session.generate_response(code)
            return response