import math
import os
import string
import re
from typing import Any

import sympy
from strands_tools import calculator
#from calculator import calculator
from strands import Agent
#from strands.models.openai import OpenAIModel
from strands_openai import OpenAIModelNonStreaming

from agentlightning import Trainer, LitAgent, NamedResources, LLM, reward, configure_logger, DevTaskLoader
from helper import extract_numbers_and_percentages

configure_logger()

# Copied and adapted from https://github.com/prompteus/calc-x/blob/master/gadgets/metrics.py


#def normalize_option(option: str) -> str:
#    """
#    >>> normalize_option("  (A)  \n")
#    'A'
#    """
#    return re.sub(r"(\s+|\(|\))", "", option)

def normalize_option(option: str) -> str:
    """
    Extracts and returns the option letter from various formats.
    
    >>> normalize_option("  (A)  \n")
    'A'
    >>> normalize_option(" (A) 2/5 \n")
    'A'
    """
    # Look for a letter inside parentheses and capture just the letter
    match = re.search(r"$\s*([A-Z])\s*$", option)
    if match:
        return match.group(1)
    # If no parentheses, try to find the first letter (fallback)
    match = re.search(r"[A-Z]", option)
    if match:
        return match.group(0)
    # Return the stripped original if no letter found
    return option.strip()

def is_option_result(result: str) -> bool:
    """
    >>> is_option_result("  A)  \n")
    True
    >>> is_option_result("  23/7 ")
    False
    """
    return normalize_option(result) in list(string.ascii_letters)


def float_eval(input_str: str) -> float:
    if " = around " in input_str:
        input_str = input_str.split(" = around ")[0]
    expr = sympy.parse_expr(input_str, evaluate=True)
    return float(expr.evalf())


def scalar_are_results_same(pred_result: str, true_result: str, rel_tol: float) -> bool:
    pred_result = str(pred_result) if pred_result is not None else ""
    true_result = str(true_result) if true_result is not None else ""

    if pred_result.strip() == true_result.strip():
        return True

    if is_option_result(true_result):
        # The task is to select correct option
        true_result = normalize_option(true_result)
        pred_result = normalize_option(pred_result)
        return pred_result == true_result

    # The task is to calculate the result as a number
    try:
        #pred_float = float_eval(pred_result)
        # further extract numbers
        pred_float = extract_numbers_and_percentages(pred_result)
        true_float = float_eval(true_result)
        return math.isclose(pred_float, true_float, rel_tol=rel_tol)
    except Exception:
        pass

    return False


@reward
async def eval(prediction: str, ground_truth: str) -> float:
    return float(scalar_are_results_same(prediction, ground_truth, 1e-2))


def get_agent(model, openai_base_url, temperature, tools):
    model_client = OpenAIModelNonStreaming(
        client_args={
            "api_key": os.environ.get("OPENAI_API_KEY", "token-abc123"),
            "base_url": openai_base_url,
        },
        # **model_config
        model_id=model,
        params={
            "temperature": temperature,
        },
        #max_tokens=8192,
    )

    calc_agent = Agent(model=model_client, tools=tools)
    return calc_agent


class CalcAgent(LitAgent):

    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        llm: LLM = resources.get("main_llm")
        #print(f"{llm=}")
        tools = [calculator]
        calc_agent = get_agent(
            llm.model,
            llm.endpoint,
            llm.sampling_parameters.get("temperature", 0.7),
            tools,
        )
        try:
            output_format = "Output the answer when you are ready. The answer should be surrounded by three sharps (`###`), in the form of ### ANSWER: <answer> ###."
            prompt = task["question"] + " " + output_format
            result = await calc_agent.invoke_async(prompt)
            #print(result.message)
            # evaluate
            answer = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", result.message["content"][-1]["text"])
            if answer:
                answer = answer.group(1)
            else:
                answer = result.message["content"][-1]["text"]
        except Exception as e:
            print("Failure:", str(e))
            answer = "None"
        reward = await eval(answer, str(task["result"]))  # reward is tracked with the decorator
        print("answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward))

    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        llm: LLM = resources.get("main_llm")
        resources = {
            "main_llm": LLM(
                endpoint=llm.endpoint,
                model=llm.model,
                sampling_parameters={"temperature": 0},
            )
        }
        return await self.training_rollout_async(task, rollout_id, resources)


if __name__ == "__main__":
    Trainer(n_workers=10).fit(CalcAgent(), "http://localhost:9999/")
    #Trainer(n_workers=1).fit(CalcAgent(), "http://localhost:9999/")
