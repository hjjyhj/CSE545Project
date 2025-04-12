import re
def main():
    judge_output = """
*** start output
No Consensus. Refer to the output from the models and rethink about the reasoning process. 
Specific Info:
<model_1> answered $70,000. Here is the reasoning process for this answer: The initial cost of the house is $80,000, and repairs cost $50,000, totaling $130,000. The value of the house increased by 150%, so the new value is $200,000. The profit is $200,000 - $130,000 = $70,000.
<model_2> answered $43,333. Here is the reasoning process for this answer: The new value of the house after repairs is $130,000. The original value before repairs is $130,000 / 1.5 = $86,667. The profit is $130,000 - $86,667 = $43,333.
<model_3> answered $120,000. Here is the reasoning process for this answer: The original value of the house is $80,000. The increase in value is 150% of $80,000, which is $120,000. The final value is $200,000. The profit is $200,000 - $80,000 = $120,000.

<model_1_original_answer_and_reasoning>: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make? Let the initial cost of the house be $C = $80,000. Let the cost of repairs be $R = $50,000. The total cost of the house is $C + R = $80,000 + $50,000 = $130,000. The value of the house increased by 150%, so the new value of the house is $V = C + 1.5C = 2.5C = 2.5($80,000) = $200,000. The profit is the difference between the new value and the total cost. Profit = $V - (C + R) = $200,000 - $130,000 = $70,000.

<model_2_original_answer_and_reasoning>: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make? ## Step 1: Calculate the new value of the house after repairs. The original price of the house was $80,000. After putting in $50,000 in repairs, the new value of the house is $80,000 + $50,000 = $130,000. ## Step 2: Calculate the original value of the house before repairs. Since the repairs increased the value of the house by 150%, the original value of the house can be found by dividing the new value by 1.5. So, the original value is $130,000 / 1.5 = $86,667. ## Step 3: Calculate the profit made by Josh. The profit made by Josh can be found by subtracting the original value of the house from the new value after repairs. So, the profit is $130,000 - $86,667 = $43,333. The final answer is: $\boxed{43333}$

<model_3_original_answer_and_reasoning>: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make? To determine the profit Josh made from flipping the house, we need to follow these steps: 
1. **Calculate the new value of the house after repairs:** 
   - The original value of the house is $80,000. 
   - Josh adds $50,000 in repairs. 
   - The new value of the house is: 80,000 + 50,000 = 130,000 
2. **Calculate the increase in value due to the 150% increase:** 
   - A 150% increase means the value increases by 150% of the original value. 
   - The increase in value is: 1.5 × 80,000 = 120,000 
3. **Calculate the final value of the house after the increase:** 
   - The final value of the house is the original value plus the increase: 80,000 + 120,000 = 200,000 
4. **Calculate the profit:** 
   - Josh buys the house for $80,000 and sells it for $200,000. 
   - The profit is: 200,000 - 80,000 = 120,000 
Therefore, the profit Josh made is $\boxed{120000}$.
*** end output
"""


    original_prompt_match = re.search(r"<model_1_original_answer_and_reasoning>:\s*(.*?)<model_2_original_answer_and_reasoning>", judge_output, re.DOTALL)
    original_prompt_1 = original_prompt_match.group(1).strip() if original_prompt_match else ""

    original_prompt_match2 = re.search(r"<model_2_original_answer_and_reasoning>:\s*(.*?)<model_3_original_answer_and_reasoning>", judge_output, re.DOTALL)
    original_prompt_2 = original_prompt_match2.group(1).strip() if original_prompt_match2 else ""

    original_prompt_match3 = re.search(r"<model_3_original_answer_and_reasoning>:\s*(.*?)\*\*\*", judge_output, re.DOTALL)
    original_prompt_3 = original_prompt_match3.group(1).strip() if original_prompt_match3 else ""
    # Extract model reasoning and answers from the structured section
    def extract_model_info(model_tag):
        pattern = rf"<{model_tag}> answered \$(.*?)\. Here is the reasoning process for this answer: (.*?)<"
        match = re.search(pattern, judge_output.replace("\n", "") + "<", re.DOTALL)
        if match:
            answer = match.group(1).strip()
            reasoning = match.group(2).strip()
            return answer, reasoning
        return "", ""

    m1_answer, m1_reasoning = extract_model_info("model_1")
    m2_answer, m2_reasoning = extract_model_info("model_2")
    m3_answer, m3_reasoning = extract_model_info("model_3")

    # Now construct updated prompts
    updated_prompts = {
        'model_1': f"""Your output: {original_prompt_1}
    Wait, let's refer to the output from the models and rethink about our initial reasoning process.
    <model_2> answered {m2_answer}. Here is the reasoning process for this answer: {m2_reasoning}
    <model_3> answered {m3_answer}. Here is the reasoning process for this answer: {m3_reasoning}
    """,
        'model_2': f"""Your output: {original_prompt_2}
    Wait, let's refer to the output from the models and rethink about our initial reasoning process.
    <model_1> answered {m1_answer}. Here is the reasoning process for this answer: {m1_reasoning}
    <model_3> answered {m3_answer}. Here is the reasoning process for this answer: {m3_reasoning}
    """,
        'model_3': f"""Your output: {original_prompt_3}
    Wait, let's refer to the output from the models and rethink about our initial reasoning process.
    <model_1> answered {m1_answer}. Here is the reasoning process for this answer: {m1_reasoning}
    <model_2> answered {m2_answer}. Here is the reasoning process for this answer: {m2_reasoning}
    """
    }
    for key, value in updated_prompts.items():
        print("output from " + key + ": " + value)
    
if __name__ == "__main__":
    main()