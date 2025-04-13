import re
import sympy
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import logging
from Utils.gsm8k_utils import create_grading_messages, query_api_chat_native
# from antlr4 import *
from math_verify import parse, verify

def last_boxed_only_string(string):
    """
    Extract the last \boxed{...}, \fbox{...}, <boxed{...}>, <boxed: ...>, <boxed number>, or <boxed>...</boxed> element from a string.
    Directly adapted from MATH dataset's implementation with additional format support.
    """
    # Attempt to find \boxed or \fbox normally first
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
    
    # If not found, attempt to correct common issues and search again
    if idx < 0:
        string = string.replace('\x08', '\\b')  # Fix \boxed
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
    
    # If standard LaTeX format is found, process it
    if idx >= 0:
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        
        if right_brace_idx is not None:
            return string[idx:right_brace_idx + 1]
    
    # If standard format not found, try alternative formats
    
    # Try <boxed>...</boxed> format first (to avoid collision with <boxed number>)
    idx = string.rfind("<boxed>")
    if idx >= 0:
        # Find the closing tag (either </boxed> or <\boxed>)
        right_tag_idx = string.find("</boxed>", idx)
        if right_tag_idx < 0:  # If standard closing tag not found, try alternative
            right_tag_idx = string.find("<\\boxed>", idx)
        
        if right_tag_idx >= 0:
            # Return the entire boxed expression including tags
            closing_tag_len = 8  # Length of "</boxed>" or "<\boxed>"
            return string[idx:right_tag_idx+closing_tag_len]
    
    # Try <boxed{...}> format
    idx = string.rfind("<boxed{")
    if idx >= 0:
        # Find the closing angle bracket
        right_bracket_idx = string.find("}>", idx)
        if right_bracket_idx >= 0:
            # Return the entire boxed expression
            return string[idx:right_bracket_idx+2]
    
    # Try <boxed: ...> format
    idx = string.rfind("<boxed:")
    if idx >= 0:
        # Find the closing angle bracket
        right_bracket_idx = string.find(">", idx)
        if right_bracket_idx >= 0:
            # Return the entire boxed expression
            return string[idx:right_bracket_idx+1]
    
    # Try <boxed number> format - check that it's not followed by </boxed>
    idx = string.rfind("<boxed ")
    if idx >= 0:
        # Find the closing angle bracket
        right_bracket_idx = string.find(">", idx)
        if right_bracket_idx >= 0:
            # Make sure this isn't part of a <boxed>...</boxed> pattern
            if string.find("</boxed>", right_bracket_idx) != right_bracket_idx + 1 and \
               string.find("<\\boxed>", right_bracket_idx) != right_bracket_idx + 1:
                # Return the entire boxed expression
                return string[idx:right_bracket_idx+1]
    
    # If no boxed format found
    return None

def clean_numbers(string):
    """
    Clean numbers in the given string by adding commas for readability.
    Adapted from MATH dataset's implementation.
    """
    if not string:
        return string
        
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        if c in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string

def extract_answer_with_pattern(text: str) -> float | None:
    """
    Extract the answer number from the response text using various patterns.
    First tries specific patterns like "The answer is: <number>".
    
    Returns:
        float: The extracted number if found
        None: If no pattern is found
    """
    if isinstance(text, dict):
        text = text.get('content') or text.get('generated_text')
    
    # Try to find number after "answer is" or "answer is:"
    if 'answer is' in text.lower():
        # Updated pattern to handle currency symbols before or after the number
        matches = re.findall(r'answer is:?\s*(?![a-zA-Z])(?:[$£€]?\s*(-?[\d,]+\.?\d*)|(-?[\d,]+\.?\d*)\s*[€])[^a-zA-Z]*', text.lower())
        if matches:
            try:
                # Get the matched number from either group and remove commas
                number = matches[0][0] or matches[0][1]  # First non-empty group
                return float(number.replace(',', ''))
            except ValueError:
                pass
    return None

def normalize_answer(answer):
    """Normalize a mathematical answer for comparison."""
    try:
        # Handle dict input
        if isinstance(answer, dict):
            answer = answer.get('content') or answer.get('generated_text') or str(answer)
        
        # Convert to string if not already
        answer = str(answer)
        
        # First try to extract boxed answer if it exists
        boxed = last_boxed_only_string(answer)
        if boxed:
            # Remove the wrapper based on format
            if boxed.startswith("\\boxed{") or boxed.startswith("\\fbox{"):
                answer = boxed[boxed.find('{')+1:boxed.rfind('}')]
            elif boxed.startswith("<boxed{"):
                answer = boxed[boxed.find('{')+1:boxed.rfind('}')]
            elif boxed.startswith("<boxed:"):
                answer = boxed[boxed.find(':')+1:boxed.rfind('>')]
            elif boxed.startswith("<boxed "):
                answer = boxed[boxed.find(' ')+1:boxed.rfind('>')]
            elif boxed.startswith("<boxed>") and (boxed.endswith("</boxed>") or boxed.endswith("<\\boxed>")):
                # Remove <boxed> and </boxed> or <\boxed>
                if boxed.endswith("</boxed>"):
                    answer = boxed[7:-8]  # Remove <boxed> and </boxed>
                else:
                    answer = boxed[7:-8]  # Remove <boxed> and <\boxed>
            logging.info(f"Boxed answer found. After removing wrapper: {answer}")
        else:
            string_match_answer = extract_answer_with_pattern(answer)
            if string_match_answer is not None:
                answer = string_match_answer
                logging.info(f"Extracted answer from pattern: {answer}")

        # Apply comprehensive LaTeX cleanup
        answer = _strip_string(answer)
        logging.info(f"After LaTeX cleanup: {answer}")
        
        # Clean numbers
        answer = clean_numbers(answer)
        logging.info(f"After cleaning numbers: {answer}")
        
        # Remove any remaining LaTeX formatting
        # answer = answer.replace('\\', '')
        # answer = answer.replace('$', '')
        
        # # Convert to lowercase and remove spaces
        # answer = answer.lower().strip()
        
        # Try to parse as mathematical expression
        try:
            try:
                # First standardize any polynomial expressions
                expr = _standardize_polynomial(answer)
                
                # # Then try direct expression parsing as it's more reliable
                # expr = parse_expr(answer)
                return str(expr)
            except Exception as e:
                logging.info(f"Direct parse failed: {e}")
                try:
                    # Only attempt LaTeX parsing if antlr4 is available
                    # Replace LaTeX fractions with division for parsing
                    latex_cleaned = answer.replace('\\frac', '')  # Remove \frac
                    latex_cleaned = re.sub(r'\{(\d+)\}\{(\d+)\}', r'(\1)/(\2)', latex_cleaned)  # Replace {a}{b} with (a)/(b)
                    expr = parse_latex(latex_cleaned)
                    return str(expr)
                except ImportError:
                    logging.info(f"LaTeX parsing failed {e}")
                    return answer
        except Exception as e2:
            logging.info(f"All parsing attempts failed: {e2}")
            # If parsing fails, return normalized string
            return answer
    except Exception as e:
        logging.error(f"Failed to normalize answer: {answer}. Error: {str(e)}")
        return answer

def extract_final_answer(response):
    """Extract the final answer from a model's response."""
    # First try to find number after "answer is" or "answer is:"
    if isinstance(response, dict):
        response = response.get('content') or response.get('generated_text')
    
    if 'answer is' in response.lower():
        # Updated pattern to handle currency symbols before or after the number
        matches = re.findall(r'answer is:?\s*(?![a-zA-Z])(?:[$£€]?\s*(-?[\d,]+\.?\d*)|(-?[\d,]+\.?\d*)\s*[€])[^a-zA-Z]*', response.lower())
        if matches:
            try:
                # Get the matched number from either group and remove commas
                number = matches[0][0] or matches[0][1]  # First non-empty group
                return number.replace(',', '')
            except ValueError:
                pass
    
    # Fallback: find any numbers and take the last one
    matches = re.findall(r'-?\d+\.?\d*', response)
    return matches[-1] if matches else None

def is_refusal(response):
    """Check if the response is a refusal to answer."""
    refusal_phrases = [
        "i apologize",
        "i cannot",
        "i'm not able",
        "i am not able",
        "unable to",
        "cannot assist",
        "cannot help",
        "won't help",
        "will not help",
        "do not assist",
        "don't assist",
        "restricted from",
    ]
    
    if isinstance(response, dict):
        response = response.get('content') or response.get('generated_text')
    
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in refusal_phrases)

def check_math_answer_with_refusal(response, correct_answer):
    """
    Check if the response matches the correct answer, accounting for refusals.
    Returns:
        1 for correct
        0 for incorrect
        -1 for refusal
    """
    if is_refusal(response):
        return -1
    
    # Check if the response appears to be truncated
    if _is_likely_truncated(response):
        logging.info("Response appears to be truncated, treating as refusal")
        return -1
    
    # Extract and normalize the model's answer
    # extracted_answer = extract_final_answer(response)
    normalized_response = normalize_answer(response)
    normalized_correct = normalize_answer(correct_answer)
    
    logging.info(f"Normalized response: {normalized_response}")
    logging.info(f"Normalized correct: {normalized_correct}")
    
    try:
        correct_parsed = parse(normalized_correct)
        response_parsed = parse(normalized_response)
        if verify(correct_parsed, response_parsed):
            return 1
    except Exception as e:
        logging.info(f"Math verification failed. Skipping to alternative methods.")

    try:
        # Try exact string match first
        if normalized_response == normalized_correct:
            logging.info("Exact string match")
            return 1
        
        # Try numerical comparison if both can be converted to floats
        try:
            response_val = float(normalized_response)
            correct_val = float(normalized_correct)
            # Allow for small numerical differences
            if abs(response_val - correct_val) < 1e-6:
                logging.info("Numerical comparison")
                return 1
        except:
            pass
        
        # Try symbolic comparison using SymPy
        try:
            response_expr = parse_expr(normalized_response)
            correct_expr = parse_expr(normalized_correct)
            if response_expr.equals(correct_expr):
                logging.info("Symbolic comparison")
                return 1
        except:
            pass
        
        logging.info("All direct matching methods failed, using LLM grading")
        # If all direct matching methods fail, use LLM grading        
        messages = create_grading_messages("check_math_answer", 
                                         f"The answer is: {normalized_correct}", 
                                         response)
        grader_response = query_api_chat_native(messages, model="gpt-4o-mini", temperature=0.0)
        llm_verdict = grader_response.lower().strip()
        
        if llm_verdict == "true":
            return 1
        elif llm_verdict == "refusal":
            return -1
        elif llm_verdict == "false":
            return 0
        else:
            return f"Grading faled with message: {llm_verdict}"
            
    except:
        return f"Error in grading"

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_sqrt(string):
    """
    Fix sqrt notation in LaTeX strings.
    Converts sqrt3 --> sqrt{3} format.
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _fix_a_slash_b(string):
    """
    Convert simple fractions using / to LaTeX \frac notation.
    For example: 3/4 -> \frac{3}{4}
    """
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    """Remove units that appear on the right side of a number."""
    # Common units to remove
    units = [
        'cm', 'mm', 'm', 'km',
        'g', 'kg', 'mg',
        'L', 'mL',
        's', 'min', 'h',
        '°', 'deg', 'rad',
        '$', '€', '£',
        'ft', 'in', 'yd', 'mi'
    ]
    
    # Sort units by length (longest first) to avoid partial matches
    units.sort(key=len, reverse=True)
    
    for unit in units:
        if string.endswith(unit):
            string = string[:-len(unit)].strip()
            break
    return string

def _standardize_polynomial(string):
    """
    Reorder polynomial terms in standard form.
    Converts expressions like '4 + x^9' to 'x^9 + 4'.
    Handles implicit multiplication (e.g., '7x' -> '7*x').
    Also standardizes the order of multiplicative factors.
    """
    try:
        # Add multiplication symbols between terms
        # Replace cases like '7x' with '7*x'
        string = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', string)
        # Replace cases like 'x(x-2)' with 'x*(x-2)'
        string = re.sub(r'([a-zA-Z0-9])\(', r'\1*(', string)
        # Replace '^' with '**' for Python-style exponentiation
        string = string.replace('^', '**')
        
        # Try to parse as sympy expression
        expr = parse_expr(string)
        
        # Expand the expression to get a canonical form
        expanded = expr.expand()
        
        # Convert back to string in standard form, replacing '**' with '^'
        return str(expanded).replace('**', '^')
    except Exception as e:
        logging.info(f"Polynomial standardization failed: {e}")
        return string

def _is_likely_truncated(response):
    """
    Check if a response is likely truncated due to token limits.
    
    Args:
        response: The model response (string or dict)
        
    Returns:
        bool: True if the response appears to be truncated, False otherwise
    """
    # Handle case where response is a dictionary
    if isinstance(response, dict):
        response_text = response.get('content') or response.get('generated_text') or str(response)
    else:
        response_text = str(response)

    # Check if the response ends with a number after "The answer is:" or similar phrases
    if response_text and re.search(r'[Tt]he answer is:?\s*-?\d+(\.\d+)?["\']*$', response_text):
        return False
    
    # Check if the response is long and ends abruptly
    if len(response_text) > 2000 and response_text[-1] not in ['.', '!', '?', ')', '}', ']', '"', "'", "$", "€", "£", "¥", ">"]:
        logging.info(f"Response appears to be truncated (length: {len(response_text)}, ending: '{response_text[-20:]}'), likely due to token limit")
        return True
    
    return False