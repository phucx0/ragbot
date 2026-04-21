from model import rag

if __name__ == "__main__":
    query = "ai là người sáng lập facebook"
    answer, docs = rag(query)

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== SOURCES ===")
    for d in docs:
        print("-", d["score"], d["text"][:200])