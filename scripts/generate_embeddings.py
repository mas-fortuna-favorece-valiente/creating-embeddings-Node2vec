"""
Script for generating gene network embeddings using Node2Vec approach
"""

import argparse
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
import gensim
import os
import sys

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate gene network embeddings using Node2Vec')
    parser.add_argument('--input', '-i', required=True, help='Input TSV file with gene interactions')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file for embeddings')
    parser.add_argument('--dimensions', '-d', type=int, default=128, help='Embedding dimensions (default: 128)')
    parser.add_argument('--walk_length', type=int, default=50, help='Random walk length (default: 50)')
    parser.add_argument('--num_walks', type=int, default=10, help='Number of walks per node (default: 10)')
    parser.add_argument('--window_size', type=int, default=10, help='Word2Vec window size (default: 10)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs (default: 20)')
    
    args = parser.parse_args()
    
    print(f"Gensim version: {gensim.__version__}")
    
    # 1. Load and process data
    print("Loading data...")
    try:
        edges = pd.read_csv(args.input, sep='\t', header=None, names=['source', 'target'])
        edges = edges.dropna().drop_duplicates()
        edges['source'] = edges['source'].astype(str)
        edges['target'] = edges['target'].astype(str)
        
        print(f"Total unique edges: {len(edges)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # 2. Create graph
    print("Creating graph...")
    try:
        graph = StellarGraph(edges=edges)
        print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    except Exception as e:
        print(f"Error creating graph: {e}")
        sys.exit(1)
    
    # 3. Generate random walks
    print("Generating random walks...")
    try:
        rw = BiasedRandomWalk(graph)
        walks = rw.run(
            nodes=list(graph.nodes()),
            length=args.walk_length,
            n=args.num_walks,
            p=1.0,
            q=1.0
        )
        print(f"Generated {len(walks)} walks")
    except Exception as e:
        print(f"Error generating walks: {e}")
        sys.exit(1)
    
    # 4. Train Word2Vec model
    print("Training Word2Vec model...")
    str_walks = [[str(node) for node in walk] for walk in walks]
    
    # Determine parameters based on gensim version
    model = Word2Vec(
            sentences=str_walks,
            vector_size=args.dimensions,
            window=args.window_size,
            min_count=1,
            sg=1,
            workers=4,
            epochs=args.epochs
        )
    
    # 5. Extract embeddings
    print("Extracting embeddings...")
    embeddings = {}
    for node_id in graph.nodes():
        embeddings[node_id] = model.wv[str(node_id)]
    
    # 6. Save results
    print(f"Saving embeddings to {args.output}...")
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')
    embeddings_df.index.name = 'gene_id'
    embeddings_df.to_csv(args.output)
    
    print(f"Embeddings saved! Shape: {embeddings_df.shape}")
    print(f"Number of genes: {len(embeddings)}")

if __name__ == "__main__":
    main()