from typing import Dict, List, Tuple, Any

class DatasetSpecificProcessing:
    """
    Base class for dataset-specific processing.
    """
    QUESTION_LITERAL = "question"
    FINAL_ANSWER_LITERAL = "answer"
    ANSWER_WITH_REASON_LITERAL = "answer_with_reason"
    TEXT_DELIMITER_PATTERN = r"<START>(.*?)<END>"
    TEXT_DELIMITER_PATTERN_MUTATION = r"<START>(.*?)<END>"
    ANSWER_START = "<ANSWER>"
    ANSWER_END = "</ANSWER>"

    def collate_to_str(self, examples: List[Dict], template: str) -> str:
        """
        Convert list of examples to string format using given template.
        """
        example_string = ""
        for example in examples:
            answer = example[self.FINAL_ANSWER_LITERAL]
            if self.ANSWER_WITH_REASON_LITERAL in example:
                answer = example[self.ANSWER_WITH_REASON_LITERAL]

            example_string += template.format(
                question=example[self.QUESTION_LITERAL],
                answer=answer
            )
        return example_string

    def access_answer(self, llm_output: str, gt_answer: str) -> Tuple[bool, str]:
        """
        Compare LLM output with ground truth answer.
        """
        return True, llm_output  # Default implementation always returns True 