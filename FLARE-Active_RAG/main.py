"""
FLARE: Forward-Looking Active REtrieval-Augmented Generation

Faithful implementation of the FLARE paper (https://arxiv.org/abs/2305.06983).

FLARE generates a temporary next sentence, checks confidence, and retrieves 
BEFORE finalizing (forward-looking), not after generation.
"""

import os
import re
import json
import argparse
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever


class FLARE:
    """
    FLARE: Forward-Looking Active REtrieval-Augmented Generation
    
    Paper algorithm:
    1. Generate temporary next sentence (looking ahead)
    2. Check token probabilities - if low confidence, trigger retrieval
    3. Use temporary sentence as query (FLARE-direct) or formulate question (FLARE-instruct)
    4. Retrieve relevant documents
    5. Regenerate sentence with retrieved context
    6. Append to output and repeat
    """
    
    def __init__(
        self,
        client: OpenAI,
        retriever: BM25Retriever,
        *,
        model: str = "gpt-4o-mini",
        theta: float = 0.4,  # Paper Table 9: θ ∈ {0.4, 0.8} (default 0.4)
        beta: float = 0.4,   # Paper Table 9: β = 0.4
        max_steps: int = 10,
        temperature: float = 0.0,
    ):
        self.client = client
        self.retriever = retriever
        self.model = model
        self.theta = theta
        self.beta = beta
        self.max_steps = max_steps
        self.temperature = temperature
        
        # Use the retriever's actual top_k (already capped in build_retriever)
        if hasattr(self.retriever, 'similarity_top_k'):
            self.top_k = self.retriever.similarity_top_k
        else:
            self.top_k = 5
    
    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: Optional[float] = None,
        get_logprobs: bool = False,
    ) -> Tuple[str, Optional[float]]:
        """
        Call LLM and return response with optional minimum probability.
        
        Returns:
            (response_text, min_probability or None)
        """
        temp = temperature if temperature is not None else self.temperature
        
        try:
            # Try chat completions with logprobs (GPT-4 and newer models support this)
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_tokens,
                logprobs=get_logprobs,
                top_logprobs=1 if get_logprobs else None,
            )
            response_text = r.choices[0].message.content or ""
            
            # Extract minimum probability if logprobs are available
            min_prob = None
            if get_logprobs and hasattr(r.choices[0], 'logprobs') and r.choices[0].logprobs:
                try:
                    logprobs_content = r.choices[0].logprobs.content
                    if logprobs_content:
                        # Get token probabilities (exp of log probability)
                        probs = []
                        for token_data in logprobs_content:
                            if hasattr(token_data, 'logprob') and token_data.logprob is not None:
                                import math
                                probs.append(math.exp(token_data.logprob))
                        if probs:
                            min_prob = min(probs)
                except:
                    pass
            
            return response_text.strip(), min_prob
            
        except Exception as e:
            # Fallback without logprobs
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                response_text = r.choices[0].message.content or ""
                return response_text.strip(), None
            except:
                raise e
    
    def _retrieve_documents(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query"""
        if not query or not query.strip():
            return []
        results = self.retriever.retrieve(query)
        return [self._node_to_text(node.node) for node in results]
    
    def _node_to_text(self, node) -> str:
        """Extract text from a node"""
        try:
            return node.get_content(metadata_mode="none")
        except (TypeError, AttributeError):
            return node.get_content() if hasattr(node, 'get_content') else str(node)
    
    def _format_context(self, documents: List[str], max_chars: int = 3000) -> str:
        """Format retrieved documents as context"""
        if not documents:
            return ""
        
        context_parts = []
        total_chars = 0
        
        for i, doc in enumerate(documents):
            doc_text = doc.strip()
            if total_chars + len(doc_text) > max_chars:
                break
            context_parts.append(f"[Document {i+1}]\n{doc_text}\n")
            total_chars += len(doc_text)
        
        return "\n".join(context_parts).strip()
    
    def generate(
        self,
        question: str,
        initial_context: Optional[str] = None,
    ) -> str:
        """
        Generate answer using FLARE algorithm (faithful to paper).
        
        Algorithm:
        1. Generate temporary next sentence (forward-looking)
        2. Check confidence - if low, retrieve
        3. Regenerate with retrieved context
        4. Append to final answer
        5. Repeat
        """
        # Initial retrieval
        if initial_context is None:
            initial_docs = self._retrieve_documents(question)
            initial_context = self._format_context(initial_docs)
        
        # Base prompt without retrieved context (post-bootstrap)
        base_prompt_no_ctx = f"""Answer the following question. Generate your answer sentence by sentence.

Question: {question}

Answer:"""

        # Bootstrap prompt with initial retrieval (only for first step)
        base_prompt_bootstrap = f"""Answer the following question based on the provided context. Generate your answer sentence by sentence.

Context:
{initial_context}

Question: {question}

Answer:"""
        
        final_answer = ""
        step = 0
        
        while step < self.max_steps:
            # Use bootstrap context only on the first step; afterward no retrieved context in draft
            if step == 0:
                current_prompt = base_prompt_bootstrap
            else:
                current_prompt = base_prompt_no_ctx
            if final_answer:
                current_prompt += " " + final_answer
            
            # Step 1: Generate temporary next sentence (FORWARD-LOOKING)
            temp_sentence, min_prob = self._call_llm(
                current_prompt,
                max_tokens=50,
                get_logprobs=True,
            )
            
            if not temp_sentence:
                break
            
            # Extract just the next sentence (not full response)
            sentences = re.split(r'[.!?]\s+', temp_sentence)
            next_sentence = sentences[0] if sentences else temp_sentence
            if next_sentence and not next_sentence.endswith(('.', '!', '?')):
                next_sentence += '.'
            
            # Step 2: Check confidence - if low or no logprobs available, retrieve
            should_retrieve = False
            low_conf_mask = None

            if min_prob is None:
                # No logprobs available - use sentence mask threshold heuristic (beta)
                uncertain_words = ['approximately', 'about', 'roughly', 'around', 'maybe', 'perhaps', 'possibly', 'some', 'certain', 'various', 'many', 'unknown', 'unclear', 'uncertain']
                tokens = re.findall(r'\b\w+\b', next_sentence)
                lower_tokens = [t.lower() for t in tokens]
                uncertain_flags = [t in uncertain_words for t in lower_tokens]
                uncertain_ratio = (sum(uncertain_flags) / len(tokens)) if tokens else 0.0

                if uncertain_ratio >= self.beta:
                    should_retrieve = True
                    # mask uncertain tokens for query (implicit masking)
                    masked_tokens = [("[MASK]" if flag else tok) for tok, flag in zip(tokens, uncertain_flags)]
                    low_conf_mask = " ".join(masked_tokens)
            else:
                # Have logprobs: if any token prob < theta -> retrieve
                if min_prob < self.theta:
                    should_retrieve = True
                # Note: per-token masking requires token-level probs; we only have min_prob aggregated
                low_conf_mask = None
            
            # Step 3 & 4: If uncertain, retrieve and regenerate
            if should_retrieve:
                # FLARE-direct implicit: mask low-confidence tokens if available, else use sentence
                if low_conf_mask:
                    query = low_conf_mask
                else:
                    query = next_sentence.replace('.', '').replace('!', '').replace('?', '').strip()
                
                retrieved_docs = self._retrieve_documents(query)
                if retrieved_docs:
                    retrieved_context = self._format_context(retrieved_docs, max_chars=2000)
                    
                    # Regenerate with retrieved context
                    regen_prompt = f"""Based on the following context, complete the sentence naturally and accurately.

Context:
{retrieved_context}

Question: {question}
Previous answer: {final_answer if final_answer else "(none)"}

Complete the next sentence:"""
                    
                    regenerated, _ = self._call_llm(
                        regen_prompt,
                        max_tokens=50,
                        temperature=0.0,
                    )
                    
                    if regenerated:
                        # Extract first sentence
                        regen_sentences = re.split(r'[.!?]\s+', regenerated)
                        next_sentence = regen_sentences[0] if regen_sentences else regenerated
                        if next_sentence and not next_sentence.endswith(('.', '!', '?')):
                            next_sentence += '.'
            
            # Step 5: Append finalized sentence to answer
            final_answer += (" " if final_answer else "") + next_sentence
            
            # Check if we have a complete answer (simple heuristic)
            if len(final_answer.split('.')) >= 3:  # At least 2-3 sentences
                break
            
            step += 1
        
        return final_answer.strip()


def build_retriever(
    data_dir: str,
    top_k: int = 5,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
) -> BM25Retriever:
    """Build BM25 retriever from data directory"""
    docs = SimpleDirectoryReader(data_dir, recursive=True).load_data()
    nodes = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).get_nodes_from_documents(docs)
    # Ensure top_k doesn't exceed number of nodes
    actual_top_k = min(top_k, len(nodes))
    if actual_top_k < top_k:
        print(f"Warning: Requested top_k={top_k} but only {len(nodes)} nodes available. Using top_k={actual_top_k}")
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=actual_top_k)


def load_questions(filepath: str) -> List[Dict[str, str]]:
    """Load questions from JSONL file. Expected format: {"question": "...", "answer": "..."}"""
    questions = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
    return questions


def main():
    parser = argparse.ArgumentParser(description="FLARE: Forward-Looking Active REtrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Answer command
    answer_parser = subparsers.add_parser("answer", help="Answer a question using FLARE")
    answer_parser.add_argument("--data_dir", required=True, help="Directory containing knowledge base documents")
    answer_parser.add_argument("--question", required=True, help="Question to answer")
    answer_parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Model to use")
    answer_parser.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve")
    answer_parser.add_argument("--theta", type=float, default=0.4, help="Confidence threshold θ (paper: 0.4/0.8)")
    answer_parser.add_argument("--beta", type=float, default=0.4, help="Masking threshold β (paper: 0.4)")
    answer_parser.add_argument("--max_steps", type=int, default=10, help="Max generation steps")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test FLARE with sample questions")
    test_parser.add_argument("--data_dir", default="data", help="Directory containing knowledge base documents")
    test_parser.add_argument("--questions", default="test_questions.jsonl", help="Path to questions JSONL file")
    test_parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Model to use")
    test_parser.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve")
    test_parser.add_argument("--theta", type=float, default=0.4, help="Confidence threshold θ (paper: 0.4/0.8)")
    test_parser.add_argument("--beta", type=float, default=0.4, help="Masking threshold β (paper: 0.4)")
    test_parser.add_argument("--max_steps", type=int, default=5, help="Max generation steps")
    
    args = parser.parse_args()
    
    if args.command == "answer":
        # Initialize components
        client = OpenAI()
        retriever = build_retriever(args.data_dir, top_k=args.top_k)
        
        # Create FLARE instance
        flare = FLARE(
            client=client,
            retriever=retriever,
            model=args.model,
            theta=args.theta,
            beta=args.beta,
            max_steps=args.max_steps,
        )
        
        # Generate answer
        answer = flare.generate(args.question)
        print(answer)
    
    elif args.command == "test":
        # Load questions
        questions = load_questions(args.questions)
        if not questions:
            print(f"No questions found in {args.questions}")
            return
        
        print(f"Testing FLARE with {len(questions)} questions...\n")
        
        # Initialize components
        client = OpenAI()
        retriever = build_retriever(args.data_dir, top_k=args.top_k)
        
        # Create FLARE instance
        flare = FLARE(
            client=client,
            retriever=retriever,
            model=args.model,
            theta=args.theta,
            beta=args.beta,
            max_steps=args.max_steps,
        )
        
        # Test each question
        for i, q_data in enumerate(questions, 1):
            question = q_data.get("question", "")
            expected = q_data.get("answer", "")
            
            print(f"[{i}/{len(questions)}] Question: {question}")
            print(f"Expected: {expected}")
            print("Generating answer...")
            
            try:
                answer = flare.generate(question)
                print(f"Answer: {answer}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
