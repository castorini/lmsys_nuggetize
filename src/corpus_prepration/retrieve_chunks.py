import argparse

from pyserini.encode import AutoQueryEncoder
from pyserini.output_writer import OutputFormat, get_output_writer
from pyserini.query_iterator import TopicsFormat, get_query_iterator
from pyserini.search.faiss import FaissSearcher
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Search a FAISS index with dense queries using Pyserini."
    )

    parser.add_argument(
        "--path-prefix",
        type=str,
        default="/mnt/users/s8sharif/search_arena",
        help="Base path for index, topics, and output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to run the encoder on (e.g., "cuda" or "cpu").',
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="BAAI/bge-m3",
        help="HuggingFace model name or path to local encoder checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of queries per search batch.",
    )
    parser.add_argument(
        "--hits", type=int, default=50, help="Number of hits to return per query."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads to use in batch search.",
    )

    args = parser.parse_args()

    # === Paths ===
    index_path = f"{args.path_prefix}/indexes/url_corpus.bge-m3"
    topics_path = f"{args.path_prefix}/collections/url_corpus/queries.tsv"
    output_path = f"{args.path_prefix}/runs/run.bge-m3.url_corpus.txt"

    # === Encoder ===
    encoder_args = {
        "encoder_dir": args.encoder,
        "device": args.device,
        "pooling": "mean",
        "l2_norm": True,
    }
    query_encoder = AutoQueryEncoder(**encoder_args)

    # === Query Iterator ===
    query_iterator = get_query_iterator(topics_path, TopicsFormat("default"))
    topics = query_iterator.topics

    # === Searcher ===
    searcher = FaissSearcher(index_path, query_encoder)

    # === Output Writer ===
    output_writer = get_output_writer(
        output_path,
        OutputFormat("trec"),
        "w",
        max_hits=args.hits,
        tag="Faiss",
        topics=topics,
    )

    # === Batch Search Loop ===
    with output_writer:
        batch_topics = []
        batch_topic_ids = []

        for index, (topic_id, text) in enumerate(
            tqdm(query_iterator, total=len(topics))
        ):
            batch_topic_ids.append(str(topic_id))
            batch_topics.append(text)

            if (index + 1) % args.batch_size == 0 or index == len(topics) - 1:
                results = searcher.batch_search(
                    batch_topics, batch_topic_ids, k=args.hits, threads=args.threads
                )
                results = [(id_, results[id_]) for id_ in batch_topic_ids]

                for topic_id, hits in results:
                    output_writer.write(topic_id, hits)

                batch_topics.clear()
                batch_topic_ids.clear()


if __name__ == "__main__":
    main()
