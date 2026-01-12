#!/usr/bin/env python3
"""
Quick test script for IRCOT implementation.
Run this to verify the setup works correctly.
"""

import os
import sys
from main import build_retriever, load_json, ircot_answer
from openai import OpenAI

def test_setup():
    """Test basic setup and data loading"""
    print("Testing IRCOT setup...")
    
    # Check data directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found!")
        return False
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not files:
        print(f"ERROR: No .txt files found in '{data_dir}'!")
        return False
    
    print(f"✓ Found {len(files)} data files in '{data_dir}'")
    
    # Check demo files
    if os.path.exists("demos_source.jsonl"):
        print("✓ Found demos_source.jsonl")
    else:
        print("⚠ demos_source.jsonl not found (optional)")
    
    if os.path.exists("dev.jsonl"):
        print("✓ Found dev.jsonl")
    else:
        print("⚠ dev.jsonl not found (optional)")
    
    # Test retriever building
    try:
        print("\nBuilding BM25 retriever...")
        retriever = build_retriever(data_dir, top_k=4)
        print("✓ Retriever built successfully")
        
        # Test retrieval
        test_query = "Einstein"
        results = retriever.retrieve(test_query)
        print(f"✓ Retrieved {len(results)} documents for query: '{test_query}'")
        
    except Exception as e:
        print(f"ERROR building retriever: {e}")
        return False
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠ OPENAI_API_KEY not set in environment")
        print("  Set it with: export OPENAI_API_KEY='your-key'")
        print("  Or the script will fail when making API calls")
    else:
        print("\n✓ OPENAI_API_KEY is set")
    
    print("\n✓ Setup test completed successfully!")
    return True


def test_answer_question():
    """Test answering a question (requires API key and demo file)"""
    print("\n" + "="*60)
    print("Testing question answering...")
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return
    
    demo_file = "demos/demos_M1_seed0.json"
    if not os.path.exists(demo_file):
        print(f"SKIP: Demo file '{demo_file}' not found")
        print("  Run: python main.py build-demos --source demos_source.jsonl --out_dir demos")
        return
    
    try:
        client = OpenAI()
        retriever = build_retriever("data", top_k=4)
        demos = load_json(demo_file)
        
        question = "What city was Albert Einstein born in?"
        print(f"\nQuestion: {question}")
        print("Processing...")
        
        answer = ircot_answer(
            Q=question,
            retriever=retriever,
            demos=demos,
            client=client,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            K=4,
        )
        
        print(f"\nAnswer: {answer}")
        print("✓ Question answering test completed!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if test_setup():
        if len(sys.argv) > 1 and sys.argv[1] == "--full":
            test_answer_question()
    else:
        print("\nSetup test failed. Please fix the issues above.")
        sys.exit(1)
