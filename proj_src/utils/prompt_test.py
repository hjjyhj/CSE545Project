import re
def main():
    judge_response= """
No Consensus. Refer to the output from the models and rethink about the reasoning process. 
Specific Info:
<model_1> answered 70000. Here is the reasoning process for this answer: Let the original value of the house be $V_0$.
Josh buys the house for $80,000, so $V_0 = 80,000$.
He puts in $50,000 in repairs.
The total cost is $80,000 + 50,000 = 130,000$.
The value of the house increased by 150%, so the new value is $V_1 = V_0 + 1.50 V_0 = 2.50 V_0$.
$V_1 = 2.50 \times 80,000 = 200,000$.
The profit is the difference between the new value and the total cost.
Profit = $V_1 - (V_0 + \text{repairs}) = 200,000 - (80,000 + 50,000) = 200,000 - 130,000 = 70,000$.

Alternatively, let the original value of the house be $V_0 = 80,000$.
He spent $50,000 on repairs.
The new value of the house is $V_1 = V_0 + \text{repairs} = 80,000 + 50,000 = 130,000$.
The value increased by 150%, so the new value is $V_1 = V_0 (1 + 1.50) = 80,000 (2.50) = 200,000$.
The profit is the difference between the new value and the original cost.
Profit = $V_1 - V_0 - \text{repairs} = 200,000 - 80,000 - 50,000 = 200,000 - 130,000 = 70,000$.

Let the original value of the house be $V_0 = 80,000$.
The repairs cost $50,000$.
The value of the house increased by 150%, so the new value is $V_1 = V_0 + 1.50 V_0 = 2.50 V_0 = 2.50(80,000) = 200,000$.
The profit is the difference between the new value and the cost of the house and repairs.
Profit = $V_1 - (V_0 + \text{repairs}) = 200,000 - (80,000 + 50,000) = 200,000 - 130,000 = 70,000$.

The value of the house increased by 150%, so the new value is $80,000 + 1.5(80,000) = 80,000 + 120,000 = 200,000$.
The cost of the house was $80,000$.
The cost of repairs was $50,000$.
The total cost was $80,000 + 50,000 = 130,000$.
The profit is $200,000 - 130,000 = 70,000$.

Final Answer: The final answer is $\boxed{70000}$

<model_2> answered -40000. Here is the reasoning process for this answer: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?  If he sells the house for $120,000, what is his return on investment?
Return on investment is the profit divided by the cost of the investment.  If the return on investment is 0.25, what is the cost of the investment?
## Step 1: Calculate the increased value of the house after repairs.
The original value of the house is $80,000. After repairs, the value increased by 150%. This means the new value is 80,000 + (150% of 80,000).

## Step 2: Calculate 150% of the original value.
150% of 80,000 is 1.5 * 80,000 = 120,000.

## Step 3: Calculate the new value of the house after repairs.
The new value is the original value plus the increase, which is 80,000 + 120,000 = 200,000.

## Step 4: Calculate the profit made by Josh.
The profit is the selling price minus the cost of the house and repairs. The cost of the house and repairs is $80,000 + $50,000 = $130,000. The profit is $120,000 - $130,000 = -$10,000.

## Step 5: Calculate the return on investment.
The return on investment is the profit divided by the cost of the investment. However, since the profit is negative, the return on investment is negative as well. The return on investment is -$10,000 / $130,000 = -0.0769 or approximately -0.08.

## Step 6: Calculate the cost of the investment.
The return on investment is given as 0.25. We can use this to find the cost of the investment. The formula for return on investment is (profit / cost) = return on investment. Rearranging the formula to solve for cost, we get cost = profit / return on investment. Plugging in the values, we get cost = -$10,000 / 0.25.

## Step 7: Calculate the cost of the investment.
The cost of the investment is -$10,000 / 0.25 = -$40,000.

The final answer is: $\boxed{-40000}$

<model_3> answered 38000. Here is the reasoning process for this answer: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make? To determine the profit Josh made from flipping the house, we need to follow these steps:

1. **Calculate the new value of the house after repairs:**
   - The original value of the house is $80,000.
   - Josh adds $50,000 in repairs.
   - Therefore, the total cost of the house after repairs is:
     \[
     80,000 + 50,000 = 130,000
     \]

2. **Determine the increase in value due to the repairs:**
   - The repairs increased the value of the house by 150%.
   - To find 150% of the original value ($80,000), we calculate:
     \[
     0.15 \times 80,000 = 12,000
     \]
   - Adding this increase to the original value gives the new value of the house:
     \[
     80,000 + 12,000 = 92,000
     \]

3. **Calculate the profit:**
   - Josh sells the house for the new value of $92,000.
   - The total cost of the house (original value plus repairs) is $130,000.
   - The profit is the difference between the selling price and the total cost:
     \[
     92,000 - 130,000 = -38,000
     \]

Since the result is negative, it indicates a loss rather than a profit. Therefore, Josh incurred a loss of:
\[
\boxed{38000}
\]

<model_1_original_answer_and_reasoning>: Final Answer: The final answer is $\boxed{70000}$
<model_2_original_answer_and_reasoning>: The final answer is: $\boxed{-40000}$
<model_3_original_answer_and_reasoning>: Since the result is negative, it indicates a loss rather than a profit. Therefore, Josh incurred a loss of:
\[
\boxed{38000}
\]
"""


    original_prompt_match = re.search(r"<model_1_original_answer_and_reasoning>:\s*(.*?)<model_2_original_answer_and_reasoning>", judge_response, re.DOTALL)
    original_prompt_1 = original_prompt_match.group(1).strip() if original_prompt_match else ""

    original_prompt_match2 = re.search(r"<model_2_original_answer_and_reasoning>:\s*(.*?)<model_3_original_answer_and_reasoning>", judge_response, re.DOTALL)
    original_prompt_2 = original_prompt_match2.group(1).strip() if original_prompt_match2 else ""

    original_prompt_match3 = re.search(r"<model_3_original_answer_and_reasoning>:\s*(.*?)\*\*\*", judge_response, re.DOTALL)
    original_prompt_3 = original_prompt_match3.group(1).strip() if original_prompt_match3 else ""
    # Extract model reasoning and answers from the structured section
    def extract_model_info_string(model_tag):
        """
        Extract answer and reasoning using string matching (not regex).
        """
        start_token = f"<{model_tag}> answered "
        reasoning_token = "Here is the reasoning process for this answer:"
        end_tag_candidates = ["<model_", "\n<model_", "\n\n<model_", "\n\n\n<model_"]  # fallback

        start_index = judge_response.find(start_token)
        if start_index == -1:
            return "", ""

        answer_start = start_index + len(start_token)
        reasoning_index = judge_response.find(reasoning_token, answer_start)
        if reasoning_index == -1:
            return "", ""

        answer = judge_response[answer_start:reasoning_index].strip()
        reasoning_start = reasoning_index + len(reasoning_token)

        # Try to find the next model section or the end
        remaining = judge_response[reasoning_start:]
        end_index = len(judge_response)
        for token in end_tag_candidates:
            idx = remaining.find(token)
            if idx != -1:
                end_index = reasoning_start + idx
                break

        reasoning = judge_response[reasoning_start:end_index].strip()
        return answer, reasoning



    m1_answer, m1_reasoning = extract_model_info_string("model_1")
    m2_answer, m2_reasoning = extract_model_info_string("model_2")
    m3_answer, m3_reasoning = extract_model_info_string("model_3")

    # Now construct updated prompts
    updated_prompt_dic = {
        'model_1': f"""
        Here is the question: original question
    These are answers generated by some students. Review each student's answer and select the answer you think is right. If you think all the answers are wrong, propose a new answer and give the reasoning process behind it.
    <model_1> answered {m1_answer}. Here is the reasoning process for this answer: {m1_reasoning}
    <model_2> answered {m2_answer}. Here is the reasoning process for this answer: {m2_reasoning}
    <model_3> answered {m3_answer}. Here is the reasoning process for this answer: {m3_reasoning}
    """,
        'model_2': f"""
        Here is the question: original question
        These are answers generated by some students. Review each student's answer and select the answer you think is right. If you think all the answers are wrong, propose a new answer and give the reasoning process behind it.
    <model_1> answered {m1_answer}. Here is the reasoning process for this answer: {m1_reasoning}
    <model_2> answered {m2_answer}. Here is the reasoning process for this answer: {m2_reasoning}
    <model_3> answered {m3_answer}. Here is the reasoning process for this answer: {m3_reasoning}
    """,
        'model_3': f"""
        Here is the question: original question
        These are answers generated by some students. Review each student's answer and select the answer you think is right. If you think all the answers are wrong, propose a new answer and give the reasoning process behind it.
    <model_1> answered {m1_answer}. Here is the reasoning process for this answer: {m1_reasoning}
    <model_2> answered {m2_answer}. Here is the reasoning process for this answer: {m2_reasoning}
    <model_3> answered {m3_answer}. Here is the reasoning process for this answer: {m3_reasoning}
    """
    }


    updated_prompt = [updated_prompt_dic[key] for key in updated_prompt_dic.keys()]
    print(updated_prompt[0])
    
if __name__ == "__main__":
    main()