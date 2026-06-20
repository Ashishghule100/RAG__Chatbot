import json
import time
import numpy as np
from src.rag.rag import RAGSystem

def load_test_questions(file="test_questions.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except:
        return [{"question": "What is the capital of France?", "expected_answer": "Paris"}]

def evaluate():
    rag = RAGSystem(model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    test_set = load_test_questions()
    latencies = []
    correct = 0
    for item in test_set:
        q = item["question"]
        start = time.time()
        result = rag.answer(q)
        latencies.append(time.time() - start)
        if item["expected_answer"].lower() in result["answer"].lower():
            correct += 1

    p95 = np.percentile(latencies, 95)
    avg = np.mean(latencies)
    acc = correct / len(test_set) if test_set else 0
    print(f"Avg latency: {avg:.2f}s, p95: {p95:.2f}s, Accuracy: {acc:.2f}")
    with open("eval_results.json", "w") as f:
        json.dump({"avg_latency": avg, "p95_latency": p95, "accuracy": acc}, f, indent=2)

if __name__ == "__main__":
    evaluate()