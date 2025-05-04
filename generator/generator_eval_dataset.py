import os
import pickle
from string import Template

from vrag.sp25.llm import init_gemini_llm, init_openai_llm
from vrag.sp25.util_text_processing import (
    extract_dict_from_text,
    read_course_transcripts,
    text_splitter,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = "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"
KEYFILE_PATH = "/home/mc76728/repo/Coargus/vrag/artifacts/sp25/cs391-project-11f0f788cfea.json"
COURSE_TRANSCRIPT_PATH = (
    "/home/mc76728/repo/Coargus/vrag/artifacts/sp25/merged_transcript.txt"
)


PROMPT_STRING = r"""
You're a Teaching Assistant for a course, provided with a segment of a professor's lecture text. Your task is to create a question and answer pair that assesses a student's understanding of the machine learning-related content. Use only the machine learning-related parts of the lecture, which include Machine Learning, Deep Learning, Computer Vision, Natural Language Processing, Optimization, Linear Algebra, Probability, and Calculus. Ignore all non-machine learning-related content.\n\nCreate a specific question and answer pair based on the machine learning-related material in the provided text. The question must be precise, directly tied to the technical content, and designed to test comprehension. Avoid vague questions, multiple-choice formats, small talk, or introductory statements about the course. Your answer should be concise and to the point. Make sure the answer is less than 3 sentences.\n\nInput:\nLecture text: $text\n\nOutput Format:\nReturn a dictionary literal string (e.g., "{'question': '...', 'answer': '...'}") with the keys:\n- question: A specific question assessing understanding of the machine learning-related material.\n- answer: The correct answer to the question, written as a concise paragraph.\n\nOutput Requirements:\n- Output ONLY the dictionary literal string, starting with { and ending with }.\n- Do not include any other text, explanations, or markdown formatting like json or .\n\nExample Output:\n{'question': 'How does gradient descent optimize a machine learning model?', 'answer': 'Gradient descent optimizes a machine learning model by iteratively adjusting the model's parameters to minimize the loss function, using the negative gradient to determine the direction and step size for each update.'}
"""


def main():
    modellist = [
        "gemini",
    ]  #  "gpt-4o-mini"
    for model in modellist:
        if model == "gemini":
            llm = init_gemini_llm(KEYFILE_PATH)
        else:
            llm = init_openai_llm(api_key=OPENAI_API_KEY)

        docs = read_course_transcripts(COURSE_TRANSCRIPT_PATH)
        chunks = text_splitter(docs, chunk_size=10000, chunk_overlap=2000)

        prompt_template = Template(PROMPT_STRING)

        # test_chunk = chunks[0:5]
        results = []
        for chunk in chunks:
            formatted_prompt = prompt_template.substitute(
                text=chunk.page_content
            )

            llm_response = llm.invoke(formatted_prompt)

            q_and_a_dict = extract_dict_from_text(llm_response.content)

            if (
                isinstance(q_and_a_dict, dict)
                and "question" in q_and_a_dict
                and "answer" in q_and_a_dict
            ):
                results.append(q_and_a_dict)

        # Define the pickle file path
        pickle_file_path = f"generator_eval_dataset_{model}.pkl"

        # Save the results list to a pickle file
        with open(pickle_file_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {pickle_file_path}")


if __name__ == "__main__":
    main()
