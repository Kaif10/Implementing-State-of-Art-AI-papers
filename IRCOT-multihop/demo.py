"""Live local multi-hop generation interleaved with BM25 retrieval."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_demo_backend import LocalDemoClient
from main import build_retriever, ircot_answer, load_json


def main():
    client = LocalDemoClient()
    root = Path(__file__).resolve().parent
    retriever = build_retriever(str(root / "data"), top_k=2)
    class TracedRetriever:
        similarity_top_k = 2
        def retrieve(self, query):
            print("RETRIEVE: "+query, flush=True)
            retriever.similarity_top_k = self.similarity_top_k
            result = retriever.retrieve(query)
            for item in result:
                print("  PASSAGE: "+str(item.node.metadata.get("file_name", item.node.node_id)), flush=True)
            return result
    question = "Which river flows through the city where Albert Einstein was born?"
    print("IRCoT / live reasoning steps interleaved with local BM25 retrieval", flush=True)
    print("QUESTION: "+question, flush=True)
    answer = ircot_answer(question, TracedRetriever(), load_json(str(root / "demos/demos_M1_seed0.json")),
                          client, client.name, K=2, max_steps=3)
    print("\nFINAL ANSWER: "+answer, flush=True)
    if "danube" not in answer.lower():
        raise SystemExit("The generated answer did not identify the Danube")
    print("DEMO COMPLETE", flush=True)


if __name__ == "__main__":
    main()
