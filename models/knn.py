import torch
import numpy as np
# https://github.com/NVlabs/RADIO?fbclid=IwZXh0bgNhZW0CMTAAAR3pFQL5yoJfk5EK5MZS3ZEHGOWX-JS81YaP7HMHbdsqJTX9y9Ra_Xf-5bo_aem_Ae7rfzbpkUMRqdPI_QFwcBv4Ilc9zT6eDuERPEb_JvqMS3B5P_IDeNCiwJgdQXyPz3D7DH2BZKq7rV1pfhj_nuQs


def knn_classify(test_image, embeddings_dict, model, K=3):
    # Flatten the dictionary to a list of (label, embedding) tuples and prepare data
    embeddings = []
    labels = []
    for label, embs in embeddings_dict.items():
        for emb in embs:

            if isinstance(emb, np.ndarray):
                emb = torch.tensor(emb, dtype=torch.float32).to('cuda')  # Convert to tensor

            embeddings.append(emb)
            labels.append(label)

    # Convert lists to tensors
    embeddings_tensor = torch.stack(embeddings)  # Shape should be [N, D]
    embeddings_tensor = embeddings_tensor.squeeze(1)
    labels_tensor = torch.tensor(labels).to('cuda')  # Shape should be [N]

    test_embedding = model(test_image)
    # Ensure test_embedding is 1xD
    if test_embedding.dim() == 1:
        test_embedding = test_embedding.unsqueeze(0)  # Shape should be [1, D]

    # Compute similarity using matrix multiplication
    try:
        # embeddings_tensor.T will have shape [D, N]
        similarity = torch.matmul(test_embedding, embeddings_tensor.T).squeeze(0)
    except RuntimeError as e:
        print(f"Dimension mismatch error: {e}")
        print(f"Test embedding shape: {test_embedding.shape}")
        print(f"Embeddings tensor shape: {embeddings_tensor.shape}")
        raise

    # Find the top K most similar embeddings
    max_sim, max_idxs = torch.topk(similarity, K)

    # Retrieve labels for the top K embeddings
    max_labels = labels_tensor[max_idxs]

    # Vote for the most common label
    predicted_label = max_labels.mode()[0].item()

    return predicted_label