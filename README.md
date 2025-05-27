# Search Arena Meets Nuggets

[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[![paper](https://img.shields.io/badge/paper-arxiv-blue.svg?style=flat)](https://arxiv.org/abs/2504.20006)

This project uses the [nuggetizer](https://github.com/castorini/nuggetizer) to evaluate LLM responses in the side-by-side battle mode.
Follow along to reproduce the evalulations for the [search-arena-v1-7k](https://huggingface.co/datasets/lmarena-ai/search-arena-v1-7k) dataset. The generated and assigned nuggets are also available in the following two newly created datasets with additional columns:
- [search-arena-v1-nuggets-5k](https://huggingface.co/datasets/castorini/search-arena-v1-nuggets-5k)
- [search-arena-v1-nuggets-with-urls-5k](https://huggingface.co/datasets/castorini/search-arena-v1-nuggets-with-urls-5k)

## 📟 Installation

### Create Conda Environment

```bash
conda create -n search_arena_nuggets python=3.10
conda activate search_arena_nuggets
```

### Install requirements
```bash
pip install -r requirements.txt
```
### Environment Setup

Create a `.env` file with your OpenAI credentials. For Azure OpenAI (default for GPT models):

```bash
AZURE_OPENAI_API_BASE=your_azure_endpoint
AZURE_OPENAI_API_ENDPOINT=your_azure_endpoint # Same as above
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_API_KEY=your_api_key
```

## Walkthrough

`nuggetize_responses.py` is the main entry point for generating and assigning nuggets.  
You can configure the sampling rate, number of worker processes, the model used for nuggetization, and more.

It supports two modes for nugget generation:

1. **Using scraped URLs from battles (when available)** in addition to LLM responses as source documents for nugget generation. In this mode, a pre-retrieved runfile in TREC format is passed to associate retrieved URL content chunks with each battle query. Additionally, a JSONL file containing the actual chunks is required.

2. **Using only LLM responses** for nugget generation. To use this mode, simply omit the retrieval-related arguments.

Example command:
```bash
PATH_PREFIX=your-path-prefix
python -m src.nuggetize_responses \
  --sampling_rate 1 \
  --path_prefix $PATH_PREFIX \
  --retrieved_runfile $PATH_PREFIX/runs/run.bge-m3.url_corpus.txt \
  --chunks_file $PATH_PREFIX/urls_chunked_corpus.jsonl
```
The generated nuggets, assignments, and aggregated results will be stored under the `nuggets/*`, `assignments/*`, and `results.jsonl` files within the specified path prefix. Additionally, `skips.json` will contain the question IDs of skipped battles along with the reasons for skipping.

To reproduce the results and analysis from the paper, run the following command from the root directory:
```bash
bash scripts/experiments.sh
```
### Corpus Preparation

Follow the pipeline below to construct a corpus from the 47K unique URLs linked to single-turn battles in [`lmarena-ai/search-arena-v1-7k`](https://huggingface.co/datasets/lmarena-ai/search-arena-v1-7k):

- `download_urls.py` downloads content from each URL, handling various formats (HTML, TXT, PDF, FTP, etc.).
- `scrape_texts.py` extracts the main textual content from the downloaded documents.
- `chunk_texts.py` splits the extracted text into overlapping chunks. In addition to being used in subsequent steps, the generated JSONL file containing these chunks will be passed to `nuggetize_responses.py` via the `--chunks_file` argument.
- `encode_urls_corpus.py` encodes the generated chunks using a multilingual encoder like `BAAI/bge-m3` and indexes them using a flat FAISS index.
- `prepare_retrieval_queries.py` formats the battle queries into a format compatible with Pyserini.
- `retrieve_chunks.py` performs dense retrieval using cosine similarity with Pyserini and FAISS to retrieve the top-k most relevant chunks per query. The output file will be in TREC eval format and passed to `nuggetize_responses.py` via the `--retrieved_runfile` argument.

To run the entire corpus preparation pipeline end-to-end, execute the following command from the root directory:

```bash
bash scripts/prepare_corpus.sh
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is built with the support of Azure's OpenAI credits.

## ✨ References

If you use this project, please cite the following relevant paper:

[[2504.20006] Chatbot Arena Meets Nuggets: Towards Explanations and Diagnostics in the Evaluation of LLM Responses](https://arxiv.org/abs/2504.20006)

```
@article{sharifymoghaddam2025chatbot,
  title={Chatbot Arena Meets Nuggets: Towards Explanations and Diagnostics in the Evaluation of LLM Responses},
  author={Sharifymoghaddam, Sahel and Upadhyay, Shivani and Thakur, Nandan and Pradeep, Ronak and Lin, Jimmy},
  journal={arXiv preprint arXiv:2504.20006},
  year={2025}
}
```
