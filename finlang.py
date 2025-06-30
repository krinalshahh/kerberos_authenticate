from sentence_transformers import SentenceTransformer, util
import torch

# Load your local FinLang model
model = SentenceTransformer('./')  # Point to your model directory

# Input data
news = "Nvidia announces new AI chips with 50% better performance"
company_statements = [
    "Nvidia company is affected by this news",
    "Apple company is affected by this news"
]

# Generate embeddings
news_embedding = model.encode(news, convert_to_tensor=True)
company_embeddings = model.encode(company_statements, convert_to_tensor=True)

# Calculate cosine similarity (relevance scores)
cosine_scores = util.cos_sim(news_embedding, company_embeddings)[0]

# Display results
print(f"News: {news}\n")
for company, score in zip(company_statements, cosine_scores):
    company_name = company.split(' company')[0]  # Extract just the company name
    print(f"Relevance to {company_name}: {score:.4f} ({(score*100):.1f}%)")

# Interpretation
print("\n[Interpretation]")
max_score, max_idx = torch.max(cosine_scores, dim=0)
most_relevant = company_statements[max_idx].split(' company')[0]
print(f"→ {most_relevant} is most affected (confidence: {max_score:.2%})")