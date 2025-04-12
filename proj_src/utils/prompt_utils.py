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
        Your final answer should be in the following format:

        Final Answer: <your answer>
        Reasoning: <detailed explanation>
        """
    else:
        prompt += """
        IMPORTANT: Imagine you are just a summarizer, and you don't have any reasoning ability. Make sure to only summarize the answer from the inputs given.
        Please analyze the candidate answers above and decide whether there is consensus among them.
        Only when there is a complete consensus amongst the models, based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
        Please do not provide the candidate answers on the answer. Do not add any reasoning of your own. Only use the output from the input given.
        Output your final answer in the following format:

        Final Answer: <your answer>
        Reasoning: <detailed explanation>

        If there is no consensus, we would like to prompt the small models again with the original question + reasoning process and answer from those models. 
        For the reasoning process, summarize them and output the most popular one from the beam search answers.
        
        After that, we would like you to output the original reasoning process of each model's most popular answer as well. 
        DO NOT shorten, compress, summarize, or paraphrase any part of the most popular original reasoning.
        YOU MUST output the **entire original reasoning process exactly as it was generated** by the model, including formatting, equations, bullet points, and line breaks.
        Even if the reasoning seems long, redundant, or includes markdown or LaTeX syntax, it must be reproduced in full.
        DO NOT replace the reasoning with a sentence like "Final answer: $20" or any compressed summary — this will be considered an incorrect output.

        Please output in the following format:

        *** start output
        No Consensus. Refer to the output from the models and rethink about the reasoning process. 
        Specific Info:
        <model_1> answered <majority_answer_from_model_1>. Here is the reasoning process for this answer: <reasoning_process_for_model_1> 
        <model_2> answered <majority_answer_from_model_2>. Here is the reasoning process for this answer: <reasoning_process_for_model_2>
        <model_3> answered <majority_answer_from_model_3>. Here is the reasoning process for this answer: <reasoning_process_for_model_3>

        <model_1_original_answer_and_reasoning>: <verbatim full original answer and reasoning from model 1>
        <model_2_original_answer_and_reasoning>: <verbatim full original answer and reasoning from model 2>
        <model_3_original_answer_and_reasoning>: <verbatim full original answer and reasoning from model 3>
        *** end output 

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
    updated_prompts = []
    for feedback in feedback_lines:
        new_prompt = original_prompt + judge_response
        updated_prompts.append(new_prompt)
        print(new_prompt)

    # # If we didn't get enough feedback for all models, duplicate the last one
    while len(updated_prompts) < len(MODEL_LIST):
        updated_prompts.append(updated_prompts[-1] if updated_prompts else original_prompt)
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
    updated_prompt_dic = {
        'model_1': f"""Your original output: {original_prompt_1}
    Wait, let's refer to the output from the models and rethink about our initial reasoning process. Your answer and the reasoning process might be wrong. Do not strictly stick to your original output and consider other models' reasoning and answers. 
    <model_2> answered {m2_answer}. Here is the reasoning process for this answer: {m2_reasoning}
    <model_3> answered {m3_answer}. Here is the reasoning process for this answer: {m3_reasoning}
    """,
        'model_2': f"""Your original output: {original_prompt_2}
    Wait, let's refer to the output from the models and rethink about our initial reasoning process. Your answer and the reasoning process might be wrong. Do not strictly stick to your original output and consider other models' reasoning and answers. 
    <model_1> answered {m1_answer}. Here is the reasoning process for this answer: {m1_reasoning}
    <model_3> answered {m3_answer}. Here is the reasoning process for this answer: {m3_reasoning}
    """,
        'model_3': f"""Your original output: {original_prompt_3}
    Wait, let's refer to the output from the models and rethink about our initial reasoning process. Your answer and the reasoning process might be wrong. Do not strictly stick to your original output and consider other models' reasoning and answers. 
    <model_1> answered {m1_answer}. Here is the reasoning process for this answer: {m1_reasoning}
    <model_2> answered {m2_answer}. Here is the reasoning process for this answer: {m2_reasoning}
    """
    }

    updated_prompt = [updated_prompt_dic[key] for key in updated_prompt_dic.keys()]
    return updated_prompts 