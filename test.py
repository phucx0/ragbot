# test.py
from model import rag, load_model

def main():
    print("Đang load model, vui lòng chờ...")
    load_model()
    print("✅ Sẵn sàng! Gõ 'exit' để thoát.\n")

    while True:
        query = input("🧑 Câu hỏi: ").strip()
        
        if not query:
            continue
        if query.lower() == "exit":
            print("Tạm biệt!")
            break

        print("⏳ Đang tìm câu trả lời...")
        answer, docs = rag(query)

        print(f"\n🤖 Trả lời: {answer}")
        print("\n📚 Nguồn tham khảo:")
        for i, d in enumerate(docs, 1):
            print(f"  [{i}] score={d['score']:.3f} | {d['text'][:150]}...")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()