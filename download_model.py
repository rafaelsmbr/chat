from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
model.save("/app/models/all-mpnet-base-v2")
print("Modelo salvo em /app/models/all-mpnet-base-v2")
