import re


def clean_response(response_text):
    """
    Clean model output to extract only the letters A, B, C, D.
    Handle cases like "Option A", "A.", "A, B", "The answer is C".
    """
    # Remove everything except letters and commas
    # Find for pattern like A, B, C, D
    # Find all uppercase letters A-D in the text
    matches = re.findall(r'\b[A-D]\b', response_text.upper())

    # If nothing is found, return empty string
    if not matches:
        return set()

    # Return set for easy comparison (the order does not matter for sets)
    return set(matches)


def calculate_score(prediction_set, golden_string):
    """
    Calculate score: 1.0 (Exact), 0.5 (Partial), 0.0 (Wrong)
    Exact: All letters are correct
    Partial: Some letters are contained in the correct set, but not all
    Wrong: There is a letter not contained in the correct set
    """
    # Clean also the golden answer (that arrives as a string: "C" o "A,B")
    golden_set = set(re.findall(r'\b[A-D]\b', golden_string.upper()))

    if not golden_set:
        print(f"Warning: Golden answer empty or malformed: {golden_string}")
        return 0.0
    if not prediction_set:
        return 0.0
    # Case 1: Equal (Exact) -> 1 point
    if prediction_set == golden_set:
        return 1.0

    # Case 2: Golden Answer CONTAIN the prediction -> 0.5 points
    # Example: Gold={A, B}, Pred={A} -> Gold contains Pred
    # Example: Gold={A}, Pred={A, B} -> Gold DOES NOT contain Pred
    elif prediction_set.issubset(golden_set):
        return 0.5

    # Case 3: Golden does not contain predicted (Wrong) -> 0 points
    else:
        return 0.0
    

def print_metrics(results, MODEL_NAME):
  perfect_answers = [r for r in results if r['score'] == 1.0]
  partial_answers = [r for r in results if r['score'] == 0.5]

  n_total = len(results)
  n_correct = len(perfect_answers)
  n_partial = len(partial_answers)
  n_wrong = n_total - n_correct - n_partial

  total_score = sum(r['score'] for r in results)

  print(f"======= {MODEL_NAME} Causal Reasoning Results =======")
  print(f"Total questions: {n_total}")
  print(f"Correct answers: {n_correct} - {n_correct/n_total*100}%")
  print(f"Partial answers: {n_partial} - {n_partial/n_total*100}%")
  print(f"Wrong answers: {n_wrong} - {n_wrong/n_total*100}%")
  print(f"Total score: {total_score}")
  print(f"Performance of the score: {total_score/n_total*100}%")
