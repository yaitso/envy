You have 1 week to complete but feel free to submit earlier. Please time how long it took you for each task. If we decide to work with you we will compensate you for your time on these tasks. 
Objective
Your objective is to create an RL task for LLM training. An RL task consists of a prompt, along with some tools and data, and a way to verify whether the task has been completed successfully. The task should teach the model a skill useful in the normal work of an ML engineer or researcher. The task should also satisfy the pass-rate requirements. We’ve provided some example tasks below. 
To get started, clone the repo here. You can follow the setup instructions in the Readme. You’ll need an Anthropic API key. We don’t expect tasks to use more than a few dollars in inference cost.
For inspiration, you can take a look at SWE_Bench_Pro, which is a collection of realistic software engineering style tasks.
Review outcomes are yes/no with limited feedback due to volume. We can usually share 1–2 high-level signals, but not detailed code review.
Requirements
The task should resemble the kinds of things an ML engineer or ML researcher might do
For each task the model must succeed between 10% and 40% of the time. You can measure this by running a task against the model at least 10 times and averaging.
The prompt must precisely encapsulate what’s verified by the grading function.
Every possible correct solution should be allowed by the grader.
For example, avoid checking for exact match against a string of code when other solutions exist.
Every requirement contained in the prompt should be checked.
For example, if the prompt asks for a dataset filtered by a certain criteria, it should be very difficult to guess the correct answer without having correctly performed filtering.
The task should teach the model something interesting and novel, or address a general weakness in the model.
There should be multiple approaches to solving the task, and the model should fail the task for a variety of reasons, and not just one reason.
The model shouldn’t fail for task-unrelated reasons like not being good at using the tools it’s given. You may need to modify the tools so that they’re suitable for the model.
The task should be concise and easy for a person to review. Using AI is totally fine, but please ensure the output is relevant and not overly verbose.
Great submissions can typically be completed in under 300 lines of code (including task instructions, grading, maybe a custom tool, or a script to download a dataset).
Example Task Ideas
(Your task doesn’t have to be any of these! This is just for illustrative purposes)
Implement a technique from an ML paper
Ask the model to write a CUDA kernel
Ask the model to clean a dataset

Job Description link https://docs.google.com/document/d/1v5VYRNXJFkVdmchXvkweBx5eOhj8FYTPt3Ucbmv17Nw/edit?tab=t.0#heading=h.inqa2vo0h4oi 

💡 Pro Tips for Candidates
To help you make the most of this assignment:
Choose an interesting and meaningful domain. Tasks that touch on areas like scientific concepts tend to perform best — they’re engaging and highlight the model’s reasoning ability.
Include tool usage. Most production tasks rely on tools (like code execution or data processing). Try to include at least one relevant tool if it makes sense for your task.
Keep it coherent. The task should be self-contained and clearly measurable. If a small sequence of subtasks (curriculum-style) helps demonstrate one coherent goal, that’s acceptable.
Aim for balance. The task should be challenging enough that the model succeeds about 10–40% of the time — not trivial, not impossible.
Be clear and precise. Make sure the grading function directly reflects what’s asked in the prompt.

