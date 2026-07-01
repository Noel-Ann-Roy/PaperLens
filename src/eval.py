

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.recommender import search


QUERIES = [
    # Natural Language Processing
    ("transformer attention mechanism", {"cs.CL", "cs.LG"}),
    ("BERT language model pretraining", {"cs.CL"}),
    ("machine translation neural", {"cs.CL"}),
    ("text classification sentiment analysis", {"cs.CL", "cs.LG"}),
    ("question answering reading comprehension", {"cs.CL"}),
    ("named entity recognition sequence labeling", {"cs.CL"}),
    ("language model GPT text generation", {"cs.CL", "cs.LG"}),
    ("summarization abstractive extractive", {"cs.CL"}),
    ("word embeddings word2vec", {"cs.CL", "cs.LG"}),
    ("dialogue systems conversational AI", {"cs.CL", "cs.AI"}),

    # Computer Vision
    ("image classification convolutional neural network", {"cs.CV"}),
    ("object detection YOLO faster RCNN", {"cs.CV"}),
    ("image segmentation semantic", {"cs.CV"}),
    ("generative adversarial network image synthesis", {"cs.CV", "cs.LG"}),
    ("face recognition deep learning", {"cs.CV"}),
    ("visual question answering multimodal", {"cs.CV", "cs.CL"}),
    ("depth estimation 3D reconstruction", {"cs.CV"}),
    ("video understanding action recognition", {"cs.CV"}),
    ("image super resolution enhancement", {"cs.CV"}),
    ("medical image analysis radiology", {"cs.CV", "cs.AI"}),

    # Machine Learning
    ("gradient descent optimization Adam SGD", {"cs.LG", "stat.ML"}),
    ("overfitting regularization dropout", {"cs.LG", "stat.ML"}),
    ("transfer learning domain adaptation", {"cs.LG", "cs.CV"}),
    ("graph neural network node classification", {"cs.LG"}),
    ("federated learning privacy", {"cs.LG"}),
    ("neural architecture search AutoML", {"cs.LG", "cs.CV"}),
    ("knowledge distillation model compression", {"cs.LG"}),
    ("meta-learning few-shot learning", {"cs.LG", "cs.AI"}),
    ("contrastive learning self-supervised", {"cs.LG", "cs.CV"}),
    ("Gaussian process Bayesian optimization", {"stat.ML", "cs.LG"}),

    # Reinforcement Learning
    ("reinforcement learning policy gradient", {"cs.LG", "cs.AI"}),
    ("Q-learning deep reinforcement learning Atari", {"cs.LG", "cs.AI"}),
    ("multi-agent reinforcement learning", {"cs.MA", "cs.AI", "cs.LG"}),
    ("model-based reinforcement learning planning", {"cs.LG", "cs.AI"}),
    ("reward shaping exploration reinforcement", {"cs.LG", "cs.AI"}),

    # Diffusion and Generative Models
    ("diffusion model score matching", {"cs.LG", "cs.CV", "stat.ML"}),
    ("variational autoencoder latent space", {"cs.LG", "stat.ML"}),
    ("normalizing flows density estimation", {"cs.LG", "stat.ML"}),
    ("image generation stable diffusion", {"cs.CV", "cs.LG"}),
    ("text to image generation DALL-E", {"cs.CV", "cs.CL"}),

    # AI Applications
    ("drug discovery molecular property prediction", {"cs.LG", "cs.AI", "q-bio"}),
    ("protein structure prediction AlphaFold", {"cs.LG", "q-bio"}),
    ("autonomous driving perception", {"cs.CV", "cs.RO", "cs.AI"}),
    ("recommendation system collaborative filtering", {"cs.IR", "cs.LG"}),
    ("anomaly detection fraud detection", {"cs.LG", "stat.ML"}),

    # Ethics and Robustness
    ("adversarial examples robustness attack", {"cs.LG", "cs.CV", "cs.CR"}),
    ("fairness bias machine learning", {"cs.LG", "cs.AI", "cs.CY"}),
    ("explainability interpretability neural network", {"cs.LG", "cs.AI"}),
    ("privacy differential privacy machine learning", {"cs.LG", "cs.CR"}),
    ("large language model alignment safety", {"cs.CL", "cs.AI", "cs.LG"}),
]


# ---------------- EVALUATION ---------------- #

def precision_at_k(results, expected_categories, k=5):
    """
    Fraction of top-k results that contain at least one expected category.
    """
    hits = 0
    for paper in results[:k]:
        paper_cats = set(paper["categories"].split())
        if paper_cats & expected_categories:
            hits += 1
    return hits / k


def run_eval():

    print("=" * 60)
    print("Research Paper Explorer — Evaluation")
    print(f"Metric: Precision@5 over {len(QUERIES)} queries")
    print("=" * 60)

    scores = []

    for query, expected in QUERIES:

        results = search(query, k=5)

        p5 = precision_at_k(results, expected, k=5)
        scores.append(p5)

        status = "✓" if p5 >= 0.6 else "✗"
        print(f"{status} [{p5:.0%}] {query}")

        if p5 < 0.6:
            print(f"       Expected: {expected}")
            for i, r in enumerate(results[:3], 1):
                print(f"       {i}. {r['categories'][:40]}  {r['title'][:60]}")

    mean_p5 = sum(scores) / len(scores)

    print()
    print("=" * 60)
    print(f"Overall Precision@5: {mean_p5:.1%}  ({sum(s==1.0 for s in scores)}/{len(scores)} perfect)")
    print("=" * 60)

    return mean_p5


if __name__ == "__main__":
    run_eval()