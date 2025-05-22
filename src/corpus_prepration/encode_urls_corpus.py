import argparse

from pyserini.encode import AutoDocumentEncoder, JsonlCollectionIterator
from pyserini.encode.optional import FaissRepresentationWriter


def main():
    parser = argparse.ArgumentParser(
        description="Encode a JSONL corpus using a dense encoder and write to FAISS index."
    )

    parser.add_argument(
        "--path-prefix",
        type=str,
        required=True,
        help="Base path for corpus and output index.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to run the encoder on (e.g., "cuda" or "cpu").',
    )
    parser.add_argument(
        "--batch-size", type=int, default=96, help="Batch size for encoding."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-m3",
        help="HuggingFace model name or path to local encoder checkpoint.",
    )
    parser.add_argument(
        "--dimension", type=int, default=1024, help="Dimensionality of the embeddings."
    )

    args = parser.parse_args()

    # Paths
    corpus_path = f"{args.path_prefix}/urls_chunked_corpus.jsonl"
    output_index_path = (
        f"{args.path_prefix}/indexes/url_corpus.{args.model_name.split('/')[-1]}"
    )

    # Encoder setup
    encoder = AutoDocumentEncoder(
        model_name=args.model_name, device=args.device, pooling="mean", l2_norm=True
    )
    embedding_writer = FaissRepresentationWriter(
        dir_path=output_index_path, dimension=args.dimension
    )
    collection_iterator = JsonlCollectionIterator(collection_path=corpus_path)

    with embedding_writer:
        for batch_info in collection_iterator(
            batch_size=args.batch_size, shard_id=0, shard_num=1
        ):
            encode_kwargs = {
                "texts": batch_info["text"],
                "fp16": False,
                "max_length": 256,
                "add_sep": False,
            }
            embeddings = encoder.encode(**encode_kwargs)
            batch_info["vector"] = embeddings
            embedding_writer.write(batch_info, fields=["text"])


if __name__ == "__main__":
    main()
