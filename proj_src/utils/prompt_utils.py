import re
from proj_src.utils.config import MODEL_LIST

def create_consensus_prompt(original_prompt, candidate_answers, is_final_iteration):
    """
    Create a prompt for the judge model to evaluate consensus among candidate answers.
    
    Args:
        original_prompt (str): The original problem statement
        candidate_answers (list): List of candidate answer dictionaries
        is_final_iteration (bool): Whether this is the final iteration
        
    Returns:
        str: The prompt for the judge model
    """
    # Base prompt with original question and candidate answers
    prompt = f"""Below are candidate answers for the question: "{original_prompt}"
Candidate Answers:
"""
    # Add each candidate answer
    for entry in candidate_answers:
        prompt += f"- {entry['output']}\n"
    
    # Add instructions based on whether it's the final iteration
    if is_final_iteration:
        
        prompt += """
        IMPORTANT: Imagine you are just a summarizer, and you don't have any reasoning ability. Make sure to only summarize the answer from the inputs given.
        Please analyze the candidate answers above and determine the most frequent final answer. 
        Do not repeat the candidate answers or the question. 
        Based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
        Please do not provide the candidate answers on the answer. If there is no consensus on the candidate answers, output the most popular answer.
        Do not add any reasoning of your own. Only use the output from the input given.
        Your response should be in the following format (Don't use $\box$ for the Final Answer, just put the number there):

        Final Conclusion:
        Reasoning: <detailed explanation>
        Final Answer: <your answer>
        """
    else:
        prompt += """
        IMPORTANT: Imagine you are just a summarizer, and you don't have any reasoning ability. Make sure to only summarize the answer from the inputs given.
        Please analyze the candidate answers above and decide whether there is consensus among them.
        Only when there is a complete consensus amongst the models, based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
        Please do not provide the candidate answers on the answer. Do not add any reasoning of your own. Only use the output from the input given.
        Output your final answer in the following format (Don't use $\box$ for the Final Answer, just put the number there):

        Final Conclusion:
        Reasoning: <detailed explanation>
        Final Answer: <your answer>

        If there is no consensus, we would like to prompt the small models again with the original question + reasoning process and answer from those models. 
        
        Please output in the following format:
        
        No Consensus. Refer to the output from the models and rethink about the reasoning process. 
        Specific Info:
        <model_1> answered <majority_answer_from_model_1>. Here is the reasoning process for this answer: <reasoning_process_for_model_1> 
        <model_2> answered <majority_answer_from_model_2>. Here is the reasoning process for this answer: <reasoning_process_for_model_2>
        <model_3> answered <majority_answer_from_model_3>. Here is the reasoning process for this answer: <reasoning_process_for_model_3>

        Note that each model can output multiple answers using beam search. For the majority_answer, give the most popular answer from each model.
        If there is a tie, give any number out of the most popular candidates. Do not add any reasoning of your own. Only use the output from the input given.
        Do not repeat the candidate answers or the question.
        You MUST follow the output format I provided to you.
        """
    
    return prompt


def update_prompts_from_feedback(original_prompt, judge_response):
    """
    Extract feedback from judge response and update prompts for next iteration.
    
    Args:
        original_prompt (str): The original problem statement
        judge_response (str): The judge's response with feedback
        
    Returns:
        list: Updated prompts for each model
    """
    # # Extract feedback using regex
    # pattern = r"(.+?)\s+answered\s+(.+?)\.\s+Other models answered\s+(.+?)\s+and\s+(.+?)\.\s+Let's check our answer again\.\s+Make sure to show your reasoning process\."
    # feedback_lines = re.findall(pattern, judge_response)
    
    # print("Feedback extracted for next iteration:")
    # for line in feedback_lines:
    #     print(line)
    
    # Create new prompts with feedback
    # updated_prompts = []
    # for feedback in feedback_lines:
    #     new_prompt = original_prompt + judge_response
    #     updated_prompts.append(new_prompt)
    #     print(new_prompt)

    # # # If we didn't get enough feedback for all models, duplicate the last one
    # while len(updated_prompts) < len(MODEL_LIST):
    #     updated_prompts.append(updated_prompts[-1] if updated_prompts else original_prompt)
    # updated_prompts = {
    #     'model_1': "",
    #     'model_2': "",
    #     'model_3': ""
    # }

    # extract the three lines w.r.t three models
    # <model_1> answered <majority_answer_from_model_1>. Here is the reasoning process for this answer: <reasoning_process_for_model_1> 
    # <model_2> answered <majority_answer_from_model_2>. Here is the reasoning process for this answer: <reasoning_process_for_model_2>
    # <model_3> answered <majority_answer_from_model_3>. Here is the reasoning process for this answer: <reasoning_process_for_model_3>
    # 
    # <model_1_most_popular> ...
    # <model_2_most_popular> ...
    # <model_3_most_popular> ...

    # TODO: implement this
    

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
    # updated_prompt_dic = {
    #     'model_1': f"""
    #     Here is the question: {original_prompt}
    #     Previous final answer from Model1: {m1_answer}, Model2: {m2_answer}, Model3: {m3_answer}
    #     Now, here is your task. Make sure to follow the instruction carefully.
        
    #     Stage 1 - Fresh attempts
    #     1. Generate a potential reasoning process for each answer.
    #     2. Mark each scratchpad with BEGIN TRY k / END TRY k (k=1‑3).
        
    #     Stage 2 - Choose the best option
    #     1. Choose the answer that you think is the right answer. Make sure to give the reason why you think is correct.
    #     2. For the other answers and the reasoning process, give short output of why you think those answers are wrong.

    #     Stage 3 - Output the final answer

    #     Think step-by-step. Take your time. Your goal is to give the most accurate and logically consistent final answer and reasoning.
    # """,
    #     'model_2': f"""
    #     Here is the question: {original_prompt}
    #     Previous final answer from Model1: {m1_answer}, Model2: {m2_answer}, Model3: {m3_answer}
    #     Now, here is your task. Make sure to follow the instruction carefully.
        
    #     Stage 1 - Fresh attempts
    #     1. Generate a potential reasoning process for each answer.
    #     2. Mark each scratchpad with BEGIN TRY k / END TRY k (k=1‑3).
        
    #     Stage 2 - Choose the best option
    #     1. Choose the answer that you think is the right answer. Make sure to give the reason why you think is correct.
    #     2. For the other answers and the reasoning process, give short output of why you think those answers are wrong.

    #     Stage 3 - Output the final answer

    #     Think step-by-step. Take your time. Your goal is to give the most accurate and logically consistent final answer and reasoning.
    # """,
    #     'model_3': f"""
    #     Here is the question: {original_prompt}
    #     Previous final answer from Model1: {m1_answer}, Model2: {m2_answer}, Model3: {m3_answer}
    #     Now, here is your task. Make sure to follow the instruction carefully.
        
    #     Stage 1 - Fresh attempts
    #     1. Generate a potential reasoning process for each answer.
    #     2. Mark each scratchpad with BEGIN TRY k / END TRY k (k=1‑3).


    #     Think step-by-step. Take your time. Your goal is to give the most accurate and logically consistent final answer and reasoning.

    # """
    # }
    updated_prompt_dic = {
        'model_1': f"""
Here is the question: {original_prompt}
These are answers generated by some students. 
<student 1> answered {m1_answer}. Here is the reasoning process for this answer: {m1_reasoning}
<student 2> answered {m2_answer}. Here is the reasoning process for this answer: {m2_reasoning}
<student 3> answered {m3_answer}. Here is the reasoning process for this answer: {m3_reasoning}
Review each student's answer and select the answer you think is right. If you think all the answers are wrong, propose a new answer and give the reasoning process behind it. Your answer:
    """,
        'model_2': f"""
Here is the question: {original_prompt}
These are answers generated by some students. 
<student 1> answered {m1_answer}. Here is the reasoning process for this answer: {m1_reasoning}
<student 2> answered {m2_answer}. Here is the reasoning process for this answer: {m2_reasoning}
<student 3> answered {m3_answer}. Here is the reasoning process for this answer: {m3_reasoning}
Review each student's answer and select the answer you think is right. If you think all the answers are wrong, propose a new answer and give the reasoning process behind it. Your answer:
    """,
        'model_3': f"""
Here is the question: {original_prompt}
These are answers generated by some students. 
<student 1> answered {m1_answer}. Here is the reasoning process for this answer: {m1_reasoning}
<student 2> answered {m2_answer}. Here is the reasoning process for this answer: {m2_reasoning}
<student 3> answered {m3_answer}. Here is the reasoning process for this answer: {m3_reasoning}
Review each student's answer and select the answer you think is right. If you think all the answers are wrong, propose a new answer and give the reasoning process behind it. Your answer:
    """     
    }


    updated_prompt = [updated_prompt_dic[key] for key in updated_prompt_dic.keys()]
    return updated_prompt