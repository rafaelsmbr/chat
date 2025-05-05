from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
model.save("./local_all_mpnet_base_v2")
print("Modelo salvo em all-mpnet-base-v2")
