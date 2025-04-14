import torch
import os
from openai import OpenAI
from google import genai
import re

# Import configuration
from proj_src.utils.config import (
    MODEL_LIST, 
    JUDGE_MODEL_NAME, 
    ORIGINAL_PROMPT, 
    K_OUTPUTS_PER_MODEL, 
    MAX_ITERATIONS
)

# Import utility functions
from proj_src.utils.model_utils import (
    load_model_and_tokenizer, 
    generate_model_outputs, 
    get_judge_evaluation
)
from proj_src.utils.prompt_utils import (
    create_consensus_prompt, 
    update_prompts_from_feedback
)


def main():
    """
    Main function that orchestrates the consensus-seeking process between multiple models.
    """
    # Initialize current prompts for each model
    current_prompts = [ORIGINAL_PROMPT] * len(MODEL_LIST)
    
    # Load judge model once at the beginning
    # judge_tokenizer, judge_model = load_model_and_tokenizer(JUDGE_MODEL_NAME)
    # judge_model = OpenAI(api_key="sk-4673fd7bbbd445f380b30ab883a43b05", base_url="https://api.deepseek.com")
    judge_model = genai.Client(api_key="AIzaSyBYcoEwGqhcFMHpkLcN9jlkLS21txReF9E")
    # Iterate until max iterations or consensus is reached
    # for iteration in range(MAX_ITERATIONS):
        # print("=" * 80)
        # print(f"Iteration {iteration + 1}: Generating candidate answers")
        # print("=" * 80)
        
        top_candidate_answers = [[{'model': 'Gemma3ForCausalLM', 'beam': 1, 'output': 'If Marcy works for the same company for 40 years, she gets an annual pension of $50,000/year. Starting after 20 years, she becomes entitled to 5% of the value of the pension per year. If she quits after 30 years, what will her annual pension be?\n\nLet $P$ be the annual pension Marcy receives.\nFor the first 20 years, Marcy receives an annual pension of $50,000/year.\nFor the next 10 years (from year 21 to year 30), she receives 5% of the value of the pension per year.\nIf she quits after 30 years, we want to find her annual pension.\n\nFor the first 20 years, Marcy receives $50,000/year.\nSo, in 20 years, she receives $20 \\times 50,000 = 1,000,000$.\nFrom year 21 to year 30, she receives 5% of the value of the pension per year.\nLet $V$ be the value of the pension at the end of 20 years.\nSince she receives $50,000/year for 20 years, the total pension is $20 \\times 50,000 = 1,000,000$.\nSo, $V = 1,000,000$.\nFor the next 10 years, she receives 5% of the value of the pension per year.\nSo, for each year from 21 to 30, she receives $0.05 \\times V = 0.05 \\times 1,000,000 = 50,000$.\nThe total pension received in these 10 years is $10 \\times 50,000 = 500,000$.\nThe total pension received over 30 years is $1,000,000 + 500,000 = 1,500,000$.\nHowever, the question asks for her annual pension if she quits after 30 years.\nFor the first 20 years, she receives $50,000/year.\nFor the next 10 years (from year 21 to year 30), she receives 5% of the value of the pension per year.\nLet $P_0$ be the initial pension, which is $50,000/year.\nAfter 20 years, the pension is $P_0 = 50,000$.\nFor the next 10 years, she receives 5% of the value of the pension per year.\nSo, the pension for year $n$ where $20 < n \\leq 30$ is $0.05 \\times P_0 = 0.05 \\times 50,000 = 2,500$.\nThe total pension for the 10 years is $10 \\times 2,500 = 25,000$.\nThe total pension received over 30 years is $20 \\times 50,000 + 10 \\times 2,500 = 1,000,000 + 25,000 = 1,025,000$.\nThe annual pension she will receive is $\\frac{1,025,000}{30} = \\frac{102,500}{3} \\approx 34,166.67$.\n\nLet $P_1 = 50,000$ be the pension for the first 20 years.\nThe total pension received in the first 20 years is $20 \\times P_1 = 20 \\times 50,000 = 1,000,000$.\nAfter 20 years, she becomes entitled to 5% of the value of the pension per year.\nLet $P_2$ be the pension for the next 10 years.\nThe value of the pension after 20 years is $P_1 = 50,000$.\nThe pension for the next 10 years is $0.05 \\times P_1 = 0.05 \\times 50,000 = 2,500$.\nThe total pension received in the next 10 years is $10 \\times 2,500 = 25,000$.\nThe total pension received over 30 years is $1,000,000 + 25,000 = 1,025,000$.\nThe annual pension she will receive is $\\frac{1,0'}]
, [{'model': 'LlamaForCausalLM', 'beam': 1, 'output': 'If Marcy works for the same company for 40 years, she gets an annual pension of $50,000/year. Starting after 20 years, she becomes entitled to 5% of the value of the pension per year. If she quits after 30 years, what will her annual pension be? \n\n## Step 1: Calculate the total pension Marcy would receive if she worked for 40 years.\nMarcy would receive $50,000 per year for 20 years, and then 5% of the value of the pension per year for the remaining 20 years. First, we need to calculate the total value of the pension she would receive in the last 20 years.\n\n## Step 2: Calculate the value of the pension Marcy would receive in the last 20 years.\nThe value of the pension in the last 20 years is 5% of the initial pension, which is $50,000. So, the value of the pension in the last 20 years is 0.05 * $50,000 = $2,500 per year.\n\n## Step 3: Calculate the total pension Marcy would receive in the last 20 years.\nSince Marcy would receive $2,500 per year for 20 years, the total pension she would receive in the last 20 years is $2,500 * 20 = $50,000.\n\n## Step 4: Calculate the total pension Marcy would receive if she worked for 40 years.\nThe total pension Marcy would receive if she worked for 40 years is the sum of the pension she would receive in the first 20 years and the pension she would receive in the last 20 years. The pension she would receive in the first 20 years is $50,000 * 20 = $1,000,000. The total pension she would receive if she worked for 40 years is $1,000,000 + $50,000 = $1,050,000.\n\n## Step 5: Calculate the annual pension Marcy would receive if she quits after 30 years.\nSince Marcy would receive $50,000 per year for the first 20 years, and then 5% of the value of the pension per year for the remaining 10 years, we need to calculate the value of the pension she would receive in the last 10 years.\n\n## Step 6: Calculate the value of the pension Marcy would receive in the last 10 years.\nThe value of the pension in the last 10 years is 5% of the initial pension, which is $50,000. So, the value of the pension in the last 10 years is 0.05 * $50,000 = $2,500 per year.\n\n## Step 7: Calculate the total pension Marcy would receive in the last 10 years.\nSince Marcy would receive $2,500 per year for 10 years, the total pension she would receive in the last 10 years is $2,500 * 10 = $25,000.\n\n## Step 8: Calculate the annual pension Marcy would receive if she quits after 30 years.\nThe annual pension Marcy would receive if she quits after 30 years is the sum of the pension she would receive in the first 20 years and the pension she would receive in the last 10 years. The pension she would receive in the first 20 years is $50,000 * 20 = $1,000,000. The total pension she would receive if she quits after 30 years is $1,000,000 + $25,000 = $1,025,000.\n\n## Step 9: Calculate the annual pension Marcy would receive if she quits after 30 years.\nSince the total pension Marcy would receive if she quits after 30 years is $1,025,000, we need to divide this amount by 30 to get the annual pension.\n\n## Step 10: Calculate the annual pension Marcy would receive if she quits after 30 years.\nThe annual pension Marcy would receive if she quits after 30 years is $1,025,000 / 30 = $34,166.67.\n\nThe final answer is: $\\boxed{$34,166.67}$'}]
, [{'model': 'Qwen2ForCausalLM', 'beam': 1, 'output': "If Marcy works for the same company for 40 years, she gets an annual pension of $50,000/year. Starting after 20 years, she becomes entitled to 5% of the value of the pension per year. If she quits after 30 years, what will her annual pension be? To determine Marcy's annual pension after 30 years of working, we need to break down the problem into two parts: the first 20 years and the next 10 years.\n\n1. **Calculate the pension for the first 20 years:**\n   - For the first 20 years, Marcy receives a fixed annual pension of $50,000.\n   - Therefore, her total pension for the first 20 years is:\n     \\[\n     20 \\times 50,000 = 1,000,000\n     \\]\n\n2. **Calculate the pension for the next 10 years:**\n   - After 20 years, Marcy becomes entitled to 5% of the value of her pension each year.\n   - The value of her pension at the start of the 21st year is $50,000.\n   - Each year, her pension increases by 5%, so the pension in the 21st year will be:\n     \\[\n     50,000 \\times 1.05 = 52,500\n     \\]\n   - In the 22nd year, her pension will be:\n     \\[\n     52,500 \\times 1.05 = 55,125\n     \\]\n   - This pattern continues, with her pension increasing by 5% each year. The pension in the \\(n\\)-th year after the 20th year (i.e., the \\(n\\)-th year overall) can be expressed as:\n     \\[\n     50,000 \\times (1.05)^{n-20}\n     \\]\n   - We need to find the sum of her pension from the 21st year to the 30th year. This is a geometric series with the first term \\(a = 52,500\\) and common ratio \\(r = 1.05\\), and there are 10 terms in this series.\n   - The sum \\(S_n\\) of the first \\(n\\) terms of a geometric series is given by:\n     \\[\n     S_n = a \\frac{r^n - 1}{r - 1}\n     \\]\n   - Substituting the values, we get:\n     \\[\n     S_{10} = 52,500 \\frac{(1.05)^{10} - 1}{1.05 - 1}\n     \\]\n   - First, calculate \\((1.05)^{10}\\):\n     \\[\n     (1.05)^{10} \\approx 1.62889\n     \\]\n   - Then, substitute back into the formula:\n     \\[\n     S_{10} = 52,500 \\frac{1.62889 - 1}{0.05} = 52,500 \\frac{0.62889}{0.05} = 52,500 \\times 12.5778 = 660,000\n     \\]\n\n3. **Calculate the total pension after 30 years:**\n   - Add the pension for the first 20 years to the pension for the next 10 years:\n     \\[\n     1,000,000 + 660,000 = 1,660,000\n     \\]\n   - Since she quits after 30 years, her annual pension will be the total pension divided by 10 years:\n     \\[\n     \\frac{1,660,000}{10} = 166,000\n     \\]\n\nTherefore, her annual pension after 30 years is \\(\\boxed{166000}\\)."}]
]
        
        # # Generate answers from each SLM
        # for model_idx, model_name in enumerate(MODEL_LIST):
        #     print("-" * 80)
        #     print(f"Generating answers from: {model_name}")
        #     print("-" * 80)
            
        #     # Load model and tokenizer
        #     tokenizer, model = load_model_and_tokenizer(model_name)
            
        #     # Generate and collect outputs
        #     model_outputs = generate_model_outputs(
        #         model, 
        #         tokenizer, 
        #         current_prompts[model_idx]
        #     )
        #     print(model_outputs)
            
        #     # Sort outputs by length (assuming longer answers might be more detailed)
        #     # sorted_outputs = sorted(
        #     #     model_outputs, 
        #     #     key=lambda output: len(output["output"]), 
        #     #     reverse=True
        #     # )
            
        #     # Keep top K outputs from this model
        #     top_candidate_answers.append(model_outputs[0])
            
        #     # Free up memory
        #     # Keep it if the memory is the issue
        #     # If you have enough memory, COMMENT IT to prevent the cost of loading the model again
        #     del model, tokenizer
        #     torch.cuda.empty_cache()
        
        
        # Create prompt for judge to evaluate consensus
        is_final_iteration = (iteration == MAX_ITERATIONS - 1)
        judge_prompt = create_consensus_prompt(
            ORIGINAL_PROMPT, 
            top_candidate_answers, 
            is_final_iteration
        )

        judge_ans = {"model1" : "", "model2": "", "model3": ""}

        summary_prompt = """You are just a summarizer. Only use the input provided to you, and do not solve or reason about the problem even if it's not correct.
        Please analyze the answer above and summarize the reasoning process for the answer derived.
        Do not repeat the candidate answer or the question. 
        Your final answer should be in the following format:

        <model> answered <answer_from_model>. Here is the reasoning process for this answer: <reasoning_process_for_model> 
        """
        for i, key in enumerate(judge_ans.keys()):
            judge_ans[key] = get_judge_evaluation(judge_model, ORIGINAL_PROMPT + top_candidate_answers[i][0]['output'] + summary_prompt)
        
        def find_ans(text):
            match = re.search(r"answered (.+?)\. Here is the reasoning process", text)

            if match:
                answer = match.group(1).strip()
                return(answer)
            assert(False)
        answers = {"model1" : find_ans(judge_ans["model"]), "model2": find_ans(judge_ans["model"]), "model3": find_ans(judge_ans["model"])}
        
        answer_string = ", ".join([f"{key}: {value}" for key, value in answers.items()])
        # Get judge's evaluation
        judge_response = get_judge_evaluation(
            judge_model, answer_string
        )

        response = ""
        for model, ans in judge_ans.items():
            response += ans + '\n'
        response += judge_response
        judge_response = response + judge_response

        # Check if we have a final answer or need another iteration
        if is_final_iteration or judge_response.strip().startswith("Final Answer:"):
            print("Final evaluation:")
            print(judge_response.strip())
            break
            
        # If no consensus, update prompts for next iteration
        if "No Consensus" in judge_response:
            
            current_prompts = update_prompts_from_feedback(
                ORIGINAL_PROMPT, 
                judge_response
            )
            print(current_prompts[0])
    
    print("=" * 80)
    print("Process completed")


if __name__ == "__main__":
    main()

