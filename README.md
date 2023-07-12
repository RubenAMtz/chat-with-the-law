# Chat with the law

This is a custom implementation of ChatGPT + LangChain + 700+ documents of the law to be accessed as an endpoint API hosted in Azure.

## Requirements
`pip install -r requirements.txt`

Overview of important libraries:  
- LangChain
- ChatGPT key
- Python 3.8+
- Azure Subscription

# Setup
1. Get embeddings through `get_embeddings.py` and place them in a folder `embeddings` at the root.
2. Install requirements.
3. Create a .env.prod file with respective keys.
4. Configure your AzureML environment (workspace, compute instances, etc.)
4. Run `deployment.py` to deploy to AzureML endpoint.

# Notes
- Embeddings were created using 512 tokens, with overlapping windows of 40 tokens, and the ADA-002 model from OpenAI.
- Pre-processing steps are kept in a different repository.
- Hybrid search consists of latent-based + keyword-based searches. Results are re-ranked using an out-of-the-box re-ranker (hosted in Azure). Hybrid search is optimized for the most accurate responses at the expense of speed.
