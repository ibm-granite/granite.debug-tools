prompt_decomposition = '''
You will be given a multi-step question. Your task is to rewrite the question so that it only asks for one specific sub-step, while providing all necessary earlier information as already completed. This isolates the sub-task for focused evaluation.

Use the provided context to frame the sub-task clearly. Preserve the original question’s tone and structure where possible, but eliminate unrelated or already-completed parts.

Example
Original Question:
Josh decides to try flipping a house. He buys a house for $80k and then puts in $50k in repairs. This increased the value of the house by 150%. How much profit did he make?

Target Step:
Increase in house value

Precomputed Info:
[
  {"segment": "Total amount Josh spent", "answer": "130K"}
]

Rewritten Question:
Josh decides to try flipping a house. He spent a total of $130k. If the value of the house increased by 150%, what is the increase in value?

Now rewrite the following question using the completed steps provided.

Reply in the following format:
Rewritten Question: <your rewritten version here>

Original Question:
{{ question }}

Target Step:
{{ specific_step_name }}

Precomputed Info:
{{ prior_info_as_list }}
'''

prompt_variation = '''
You will be given a question along with a partially completed step-by-step solution.
Your task is to rewrite the original question by incorporating the parts of the solution that have already been completed, so the rewritten question reflects that those steps are done. The new question should only require solving for the remaining steps. When rewriting, preserve the structure and wording of the original question as much as possible—only revise or replace the parts that are directly affected by the completed steps. Use the completed work to inform the new phrasing, as if someone is picking up the problem mid-way with that progress already understood. The rewritten question must be a single line without any line breaks.

Example
Original Question:
Josh decides to try flipping a house. He buys a house for $80k and then puts in $50k in repairs. This increased the value of the house by 150%. How much profit did he make?

Solved Segments:
[
  {"segment": "Calculate total amount Josh spent", "answer": "130K"},
  {"segment": "Calculate increase in house value", "answer": "120K"},
]

Rewritten Question:
Josh decides to try flipping a house. He spent a total of $130k for the house and the repairs. This increased the value of the house by $120K. When the original value of the house was $80k, how much profit did he make?

Now rewrite the following question using the completed steps provided. Reply in the following format:
Rewritten Question: <your rewritten version here>

Original Question:
{{ question }}

Solved Segments:
{{ solved_sequential_segments }}
'''

prompt_correctness_value = '''
You are given a ground truth answer and a model answer.
Your task is to decide whether the two answers are equivalent in value.

**Rating Guidelines**
1. Ignore formatting differences (e.g., 2, "2", 2.0, answer: 2 should all be treated as the same).
2. Treat numbers written as words (e.g., two, forty-five) as equivalent to their numeric forms.
3. Units must be considered: 2 kg is not equal to 2 g, but 2000 g is equal to 2 kg.
4. Time expressions should be normalized: treat equivalent times as the same value even if expressed differently (e.g., "7:00", "7 am", "07:00", or "7 o’clock" all mean the same; "1 pm–3 pm" = "13:00–15:00"). Overlapping time intervals must match in value, regardless of format.
5. If either the model output or ground-truth answer is empty, missing, or unspecified, return a score of 0 regardless of other conditions.

**Model Output** 
{{validator_final_answer}}

**Ground-truth Answer** 
{{ground_truth}}

**Scoring Criteria**
* Score 1: If the answers are equivalent in value.
* Score 0: If the answers are different in value.

Return your rating as a json of the form {"score": your_score, "justification": your_justification}. No additional explanation, text, or formatting outside the JSON.
'''


prompt_sequential_segments_answers_reasoning = '''
You will be given a question and step-by-step sequential segments that detail the reasoning or actions needed to answer the question. Your task is to solve each segment in order. For each segment, provide a clear step-by-step reasoning and the final result for that segment.
Your output must be a list of answers where each corresponds to each segement in the following JSON format:

[
  {"explanation": "[detailed reasoning for the first segment]", "answer": "[final answer to the first segment]"},
  {"explanation": "[detailed reasoning for the second segment]", "answer": "[final answer to the second segment]"},
  ...
]


Example:
Josh decides to try flipping a house. He buys a house for $80k and then puts in $50k in repairs. This increased the value of the house by 150%. How much profit did he make? Format your answer as a JSON, like JSON = {\"explanation\": <your step by step solution>, \"answer\": <final answer>}.
[
  {"segment": "Total amount Josh spent"},
  {"segment": "Increase in house value"},
  {"segment": "New value of the house"},
  {"segment": "Profit"},
  {"segment": "Format the final result as a JSON"}
]
Output:
[
  {"explanation": "Josh spent $80k to buy the house and $50k on repairs. Adding them gives 80k + 50k = 130k.", "answer": "130k"},
  {"explanation": "The original value was 80k. A 150% increase means 1.5 × 80k = 120k.", "answer": "120k"},
  {"explanation": "The new value is original value + increase = 80k + 120k = 200k.", "answer": "200k"},
  {"explanation": "Profit = New value - Total spent = 200k - 130k = 70k.", "answer": "70k"}
  {"explanation": "Format the final result as a JSON.", "answer": "{\"explanation\": \"Josh spent 80k to buy the house and 50k on repairs, totaling 130k. The house increased by 150% of the original 80k, which is 120k. The new value of the house is 80k + 120k = 200k. Profit is new value minus total spent, 200k - 130k = 70k.\", \"answer\": \"70k\"}"}
]

Now complete the task for the following question:
{{ question }}
{{ sequential_segments }}

Only output the JSON array, strictly following the format.
'''
prompt_segment = '''
You will be given a question that requires multiple reasoning or computational steps.
Your task is to break down the instruction into explicit, step-by-step sequential segments that detail the actions needed to answer the question. Each segment should represent a distinct actionable operation.

Rules:
- Each segment must represent a concrete reasoning or computational action.
- Each segment must yield a concrete intermediate result.
- Each segment must be non-overlapping and cover all parts of the instruction.
- Each segment must represent a single unit of instruction. If a step contains multiple actions (e.g., "add and divide"), split it into separate segments.
- Each segment must directly involve computation or logical derivation.
- Each segment only describes the action not the solution.
- The last segment must represent the final step such that the response to this segment would be the final answer.
- Do not include numeric or textual answers inside the segments. Only describe the action needed to obtain them.
- Limit the total number of segments to 6 or fewer.

Your output must be a list of these segments in the following JSON format:
[
  {"segment": "[first step]"},
  {"segment": "[second step]"},
  ...
]

Here are some examples:

Example Query:
Josh decides to try flipping a house. He buys a house for $80k and then puts in $50k in repairs. This increased the value of the house by 150%. How much profit did he make? 

Example Output:
[
  {"segment": "Add the purchase price and repair costs to find total spending"},
  {"segment": "Calculate the increase in house value using 150%"},
  {"segment": "Find the new value of the house after the increase"},
  {"segment": "Subtract total spending from the new value to get the profit"}
]

Example Query:
It takes Sarah an average of 15 minutes and 36 seconds to solve 2 puzzles. If she wants to solve 12 puzzles at the same rate, it will take her X hours, Y minutes, and Z seconds. 

Example Output:
[
  {"segment": "find the time it takes to solve one puzzle"},
  {"segment": "Multiply the time for one puzzle by 12 to get the total time for 12 puzzles"},
  {"segment": "Convert the total seconds into hours, minutes, and seconds"}
]

Example Query:
In a movie, the character starts a workout at 17:00 (24hr). He warms up for 13 hours, 10 minutes and 49 seconds, does cardio for 9 hours, 7 minutes and 16 seconds, strength training for 22 hours, 10 minutes and 55 seconds, and cool down for 8 hours, 15 minutes and 10 seconds. What time does he finish his workout?

Example Output:
[
  {"segment": "Add all workout durations together to get total workout time"},
  {"segment": "Add the total workout time to the start time to get the end time in total seconds"},
  {"segment": "Convert the total end time seconds back into days, hours, minutes, and seconds"}
]

Example Query:
The conference began at 09:30:00 AM and concluded at 04:45:00 PM. Calculate the total elapsed time during the conference.

Example Output: 
[
  {"segment": "count duration from 09:30:00 to 12:00:00 (midday)"},
  {"segment": "count duration from 12:00:00 (midday) to 16:45:00"},
  {"segment": "Final elapsed hours, minutes, and seconds"}
]
Example Query:
Mia is available from 9:00 to 11:00 and from 1:00 to 3:00, while Lucas is available from 10:00 to 12:00 and from 2:00 to 4:00. How many 1-hour meeting slots are possible if the meeting must start on the hour or half hour?

Example Output:
[
  {"segment": "Find the overlapping time slots between Mia and Lucas"},
  {"segment": "Determine the possible 1-hour meeting slots within the overlapping time slots that start on the hour or half hour"},
  {"segment": "Count the total number of possible 1-hour meeting slots"}
]

Now complete the task for the following question:
{{ question }}
'''