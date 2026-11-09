"""A small live demonstration of the existing generate/execute/grade pipeline."""
import argparse
from program_of_thought import PoTModel, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--question", default="There are 3 boxes with 4 pencils each. How many pencils are there in total?")
    parser.add_argument("--gold", default="12")
    args = parser.parse_args()
    print("Program of Thoughts / live generation, Python execution and exact numeric grading", flush=True)
    print(f"Question: {args.question}\nExpected answer: {args.gold}", flush=True)
    model = PoTModel(args.model, temperature=0, max_new_tokens=160)
    result = solve(model, args.question, args.gold, exec_timeout=5)
    print(f"\nGENERATED PROGRAM:\n{result.code}\n", flush=True)
    print(f"Interpreter output: {result.pred}\nExecution: {result.exec_status}\nCorrect: {result.correct}", flush=True)
    if not result.correct:
        raise SystemExit("The generated solution did not match the expected answer")
    print("DEMO COMPLETE", flush=True)


if __name__ == "__main__":
    main()
