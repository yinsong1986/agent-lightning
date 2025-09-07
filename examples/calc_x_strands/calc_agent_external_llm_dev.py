import os
from agentlightning import Trainer, DevTaskLoader, LLM
from calc_agent_external_llm import CalcAgent


def dev_task_loader() -> DevTaskLoader:
    return DevTaskLoader(
        tasks=[
            {
                "question": "What is 2 + 2?",
                "result": "4",
            },
            {
                "question": "What is 3 * 5?",
                "result": "15",
            },
            {
                "question": "What is the square root of 16?",
                "result": "4",
            },
            {
                "question": "Jim has a 20 pack of gum.  He chews 1 piece of gum for every 2 hours he\'s at school over a school day that lasts 8 hours.  He chews 1 piece on the way home from school and 1 stick after dinner.  He also gives half the gum he has remaining to his sister when she asks for some right before bed.  How many pieces of gum does Jim have left at the end of the day?",
                "result":"7",
            },
            {
                "question":"Lewis earns $ 2 every week during the harvest. If he earns a total of $ 178, how many weeks did the harvest last?",
                "result":"89",
            },
            {
                "question":"The number of people in workshop A is the total number of people in workshops A and B (7\\/12). Now 20 people are transferred from workshop A to workshop B. At this time, the ratio of the number of people in workshops A and B is 5:7. What is the original number of workshops in A? people?","result":"70"},{"question":"Jim spends 2 hours watching TV and then decides to go to bed and reads for half as long.  He does this 3 times a week.  How many hours does he spend on TV and reading in 4 weeks?",
                "result":"36",
            },
            {
                "question":"Grandpa is 72 years old this year. Dad\'s age is grandpa\'s (4\\/9). My age is that of my father (3\\/8). So how old am I this year?",
                "result":"12",
            },
        ],
        resources={
            "main_llm": LLM(
                endpoint="", model="", sampling_parameters={"temperature": 0.0}
            ),
        },
    )


if __name__ == "__main__":
    Trainer(n_workers=1, dev=True, max_tasks=7).fit(CalcAgent(), "http://localhost:9999/", dev_task_loader())
