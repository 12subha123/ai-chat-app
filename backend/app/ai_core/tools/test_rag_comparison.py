from ..knowledge.rag_context import RAGContextBuilder
from .llm_tool import generate_llm_response
from .rag_llm_tool import RAGLLMTool


QUESTIONS = [
    "What is gravity?",
    "What causes gravitational acceleration near Earth?",
    "What are Newton's laws?",
    "What is the relationship between force and motion?",
    "How does general relativity describe gravity?",
]


def run_comparison():
    rag_tool = RAGLLMTool(
        context_builder=RAGContextBuilder(top_k=2)
    )

    print("=" * 70)
    print("BASELINE LLM VS RAG COMPARISON")
    print("=" * 70)

    for index, question in enumerate(QUESTIONS, start=1):
        print()
        print(f"QUESTION {index}: {question}")
        print("-" * 70)

        baseline = generate_llm_response(question)

        rag_response = rag_tool.generate(question)

        print("BASELINE:")
        print(baseline)

        print()
        print("RAG:")
        print(rag_response)

        print("-" * 70)


if __name__ == "__main__":
    run_comparison()