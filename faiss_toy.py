from ml_util.modelling.faiss_interface import IndexWrapper, KMeans

from ml_util.modelling.sentence_transformer_interface import  SentenceTransformerHolder

st_holder = SentenceTransformerHolder.create("pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb")

sents = [
    "Using this method, we would take a query vector xq, identify the cell it belongs to, and then use our IndexFlatL2 (or another metric) to search between the query vector and all other vectors belonging to that specific cell.",
    "So, we are reducing the scope of our search, producing an approximate answer, rather than exact (as produced through exhaustive search).",
    "To implement this, we first initialize our index using IndexFlatL2 — but this time, we are using the L2 index as a quantizer step — which we feed into the partitioning IndexIVFFlat index."
    "Here we’ve added a new parameter nlist. We use nlist to specify how many partitions (Voronoi cells) we’d like our index to have.",
    "Now, when we built the previous IndexFlatL2-only index, we didn’t need to train the index as no grouping/transformations were required to build the index.",
    "Because we added clustering with IndexIVFFlat, this is no longer the case.",
    "Just one more added sentence."
]

embeddings = st_holder.encode(sents)

# index_wrapper = IndexWrapper(embeddings.copy(), labels=None)
# distances, indices = index_wrapper.flat_search()
# #distances = np.sqrt(distances)
# print(f"distances 0, 1: {distances[0, 1]}")

kmeans = KMeans(embeddings.copy(),
                labels=None,
                label_batch_size=2)




pass



