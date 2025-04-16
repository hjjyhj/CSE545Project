import json
import re

def extract_answer_solution(json_str):
    """
    Extract the 'answer' and 'solution' fields from a JSON string,
    even if wrapped in markdown fences. Supports numeric or string
    for 'answer' and string for 'solution'. Falls back to regex if
    parsing fails.
    
    Args:
        json_str (str): A JSON‐formatted string, possibly with ```json ... ``` fencing.
    
    Returns:
        tuple: (answer: int|float|str, solution: str)
    
    Raises:
        ValueError: If required fields cannot be found or parsed.
    """
    # 1. Strip markdown fences if present
    s = json_str.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    
    # 2. Try normal JSON parsing
    try:
        data = json.loads(s)
        if 'answer' not in data or 'solution' not in data:
            return "",""
        return data['answer'], data['solution']
    
    except (json.JSONDecodeError, ValueError):
        # 3. Fallback via regex
        # answer: either "..." or number
        ans_re = re.search(
            r'"answer"\s*:\s*(?:"([^"]*)"|([0-9]+(?:\.[0-9]+)?))',
            s
        )
        # solution: must be a quoted string
        sol_re = re.search(
            r'"solution"\s*:\s*"([^"]*)"',
            s
        )
        if not ans_re or not sol_re:
            return "",""
        
        # parse answer
        if ans_re.group(1) is not None:
            answer_val = ans_re.group(1)
        else:
            num = float(ans_re.group(2))
            answer_val = int(num) if num.is_integer() else num
        
        # parse solution
        solution_val = sol_re.group(1)
        
        return answer_val, solution_val

a
def extract_consensus_final(json_str):
    """
    Extract the 'consensus' and 'final_answer' fields from a JSON string,
    even if wrapped in markdown fences. Supports boolean, numeric, or string
    values for both fields. Falls back to regex if parsing fails.
    
    Args:
        json_str (str): A JSON-formatted string, possibly with ```json ... ``` fencing.
    
    Returns:
        tuple: (consensus: bool, final_answer: int|float|str)
    
    Raises:
        ValueError: If required fields cannot be found or parsed.
    """
    # 1. Strip markdown fences if present
    s = json_str.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    
    # 2. Try normal JSON parsing
    try:
        data = json.loads(s)
        if 'consensus' not in data or 'final_answer' not in data:
            return False,""
        
        consensus_val = str(data['consensus']).lower() == 'true'
        return consensus_val, data['final_answer']
    
    except (json.JSONDecodeError, ValueError):
        # 3. Fallback via regex
        # consensus: either "..." or true/false
        cons_re = re.search(
            r'"consensus"\s*:\s*(?:"([^"]*)"|(\btrue\b|\bfalse\b))',
            s, re.IGNORECASE
        )
        # final_answer: either "..." or number
        fa_re = re.search(
            r'"final_answer"\s*:\s*(?:"([^"]*)"|([0-9]+(?:\.[0-9]+)?))',
            s
        )
        if not cons_re or not fa_re:
            return False,""
        
        # parse consensus
        if cons_re.group(1) is not None:
            consensus_val = cons_re.group(1)
        else:
            consensus_val = cons_re.group(2).lower() == 'true'
        
        # parse final_answer
        if fa_re.group(1) is not None:
            final_answer_val = fa_re.group(1)
        else:
            # numeric string → float or int
            num = float(fa_re.group(2))
            final_answer_val = int(num) if num.is_integer() else num
        
        return consensus_val, final_answer_val
