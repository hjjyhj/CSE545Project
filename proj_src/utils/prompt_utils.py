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
        Please analyze the candidate answers above and determine the most frequent final answer. 
        Do not repeat the candidate answers or the question. 
        Based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
        Please do not provide the candidate answers on the answer.
        Your final answer should be in the following format:

        Final Answer: <your answer>
        Reasoning: <detailed explanation>
        """
    else:
        prompt += """
        Please analyze the candidate answers above and decide whether there is consensus among them.
        If there is consensus, based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
        Please do not provide the candidate answers on the answer.
        Output your final answer in the following format:

        Final Answer: <your answer>
        Reasoning: <detailed explanation>

        If there is no consensus, output in the following format:

        No Consensus. 
        Specific Info:
        <model_name_1> answered <majority_answer1>. Other models answered <majority_answer2> and <majority_answer3>. Let's check our answer again. Make sure to show your reasoning process.
        <model_name_2> answered <majority_answer2>. Other models answered <majority_answer1> and <majority_answer3>. Let's check our answer again. Make sure to show your reasoning process.
        <model_name_3> answered <majority_answer3> Other models answered <majority_answer1> and <majority_answer2>. Let's check our answer again. Make sure to show your reasoning process.

        Note that each model can output multiple answers using beam search. For the majority_answer, give the most popular answer from each model.
        If there is a tie, give any number out of the most popular candidates.
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
    # Extract feedback using regex
    pattern = r"(.+?)\s+answered\s+(.+?)\.\s+Other models answered\s+(.+?)\s+and\s+(.+?)\.\s+Let's check our answer again\.\s+Make sure to show your reasoning process\."
    feedback_lines = re.findall(pattern, judge_response)
    
    print("Feedback extracted for next iteration:")
    for line in feedback_lines:
        print(line)
    
    # Create new prompts with feedback
    updated_prompts = []
    for feedback in feedback_lines:
        new_prompt = original_prompt + " ".join(feedback)
        updated_prompts.append(new_prompt)
    
    # If we didn't get enough feedback for all models, duplicate the last one
    while len(updated_prompts) < len(MODEL_LIST):
        updated_prompts.append(updated_prompts[-1] if updated_prompts else original_prompt)
    
    return updated_prompts 