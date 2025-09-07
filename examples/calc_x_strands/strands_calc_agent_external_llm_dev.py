import os
from agentlightning import Trainer, DevTaskLoader, LLM
from examples.calc_x.strands_calc_agent_external_llm import CalcAgent


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
                "question":"In how many ways can a teacher in a kindergarten school arrange a group of 3 children (Susan, Tim and Zen) on 3 identical chairs in a straight line so that Susan is on the left of Tim?\tPick: A) 7 B) 3 C) 2 D) 1 E) 6",
                "result":"B",
            },
            {
                "question":"The angle between the two radii of a sector is 60°, and its area is ((())/(())) of the area of the circle on which it is located.",
                "result":"1/6",
            },
            {
                "question": "rectangular nursery, 40 meters long and 18 meters wide, is calculated as 5 seedlings per square meter. How many saplings can this nursery raise?",
                "result": "3_600",
            },
        ],
        resources={
            "main_llm": LLM(
                endpoint="", model="", sampling_parameters={"temperature": 0.0}
            ),
        },
    )


if __name__ == "__main__":
    Trainer(n_workers=1, dev=True, max_tasks=8).fit(CalcAgent(), "http://localhost:9999/", dev_task_loader())
