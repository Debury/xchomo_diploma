"""
CLI for climate embeddings generation and RAG queries.

Commands:
  generate  - Generate embeddings from climate data file
  build-index - Build vector index from embeddings
  ask - Ask a question using RAG
"""

import argparse
import json
import sys
from pathlib import Path

def cmd_generate(args):
    """Generate embeddings from a climate data file."""
    from climate_embeddings.loaders import load_raster_auto, raster_to_embeddings, save_embeddings
    
    print(f"Loading {args.input}...")
    result = load_raster_auto(
        args.input,
        chunks="auto",
        variables=args.variables.split(",") if args.variables else None,
    )
    
    print(f"Detected format: {result.metadata['format']}")
    print(f"Generating embeddings...")
    
    embeddings = raster_to_embeddings(
        result,
        normalization=args.normalization,
    )
    
    print(f"Generated {len(embeddings)} embeddings")
    
    save_embeddings(embeddings, args.output, fmt="jsonl")
    print(f"Saved to {args.output}")


def cmd_build_index(args):
    """Build vector index from embeddings JSONL file."""
    from climate_embeddings.rag import build_index_from_embeddings
    
    print(f"Building index from {args.embeddings}...")
    index = build_index_from_embeddings(
        args.embeddings,
        dim=args.dim,
        metric=args.metric,
    )
    
    print(f"Index built with {len(index)} vectors")
    
    index.save(args.output)
    print(f"Saved index to {args.output}")


def cmd_ask(args):
    """Ask a question using RAG."""
    from climate_embeddings.index import VectorIndex
    from climate_embeddings.embeddings import get_text_embedder
    from climate_embeddings.rag import RAGPipeline
    from src.llm.ollama_client import OllamaClient
    
    print(f"Loading index from {args.index}...")
    index = VectorIndex.load(args.index)
    
    print(f"Loading text embedder ({args.model})...")
    text_embedder = get_text_embedder(args.model)
    
    print(f"Initializing LLM...")
    llm = OllamaClient()
    
    rag = RAGPipeline(
        index=index,
        text_embedder=text_embedder,
        llm_client=llm,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    
    print(f"\nQuestion: {args.question}")
    print("Retrieving context and generating answer...\n")
    
    answer = rag.ask(args.question)
    
    print(f"Answer:\n{answer}")


def main():
    parser = argparse.ArgumentParser(
        description="Climate Embeddings CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    parser_gen = subparsers.add_parser("generate", help="Generate embeddings from climate data")
    parser_gen.add_argument("input", help="Input file path")
    parser_gen.add_argument("-o", "--output", default="embeddings.jsonl", help="Output JSONL file")
    parser_gen.add_argument("--variables", help="Comma-separated variable names")
    parser_gen.add_argument("--normalization", default="zscore", choices=["zscore", "minmax", "none"])
    
    # Build-index command
    parser_idx = subparsers.add_parser("build-index", help="Build vector index from embeddings")
    parser_idx.add_argument("embeddings", help="Input embeddings JSONL file")
    parser_idx.add_argument("-o", "--output", default="index.pkl", help="Output index file")
    parser_idx.add_argument("--dim", type=int, required=True, help="Embedding dimension")
    parser_idx.add_argument("--metric", default="cosine", choices=["cosine", "dot", "euclidean"])
    
    # Ask command
    parser_ask = subparsers.add_parser("ask", help="Ask a question using RAG")
    parser_ask.add_argument("question", help="Question to ask")
    parser_ask.add_argument("--index", required=True, help="Path to vector index file")
    parser_ask.add_argument("--model", default="minilm", help="Text embedding model")
    parser_ask.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser_ask.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "generate":
            cmd_generate(args)
        elif args.command == "build-index":
            cmd_build_index(args)
        elif args.command == "ask":
            cmd_ask(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
