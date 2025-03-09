import torch
import torch.nn.functional as F

from tqdm import tqdm
class SelfAttentionClusteringPredictor:
    """
    A predictor that uses self-attention mechanisms to compute reviewer embeddings.
    
    This predictor computes a cosine similarity self-attention matrix for each reviewer's
    publications, then uses the attention weights to compute a weighted sum of the
    publication embeddings, resulting in a more representative reviewer embedding.
    """
    
    def __init__(self, top_k=None, batch_size=32, device=None):
        """
        Initialize the self-attention clustering predictor.
        
        Args:
            top_k: Number of top attention weights to keep (None means keep all)
            batch_size: Batch size for processing reviewers in parallel
            device: Device to use for computation (None for automatic detection)
        """
        self.top_k = top_k
        self.batch_size = batch_size
        
        # Set device automatically if not provided
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
    
    def _compute_reviewer_embedding(self, publication_embeddings):
        """
        Compute reviewer embedding using self-attention mechanism.
        
        Args:
            publication_embeddings: Tensor of shape (n_publications, embedding_dim) containing
                                   the embeddings of a reviewer's publications
        
        Returns:
            Tensor of shape (embedding_dim,) representing the reviewer's embedding
        """
        # Handle case with single publication
        if publication_embeddings.shape[0] == 1:
            return publication_embeddings.squeeze(0)
        
        # Compute cosine similarity matrix
        # Normalize embeddings for cosine similarity
        normalized_embeddings = F.normalize(publication_embeddings, p=2, dim=1)
        
        # Compute attention scores (dot product of normalized vectors = cosine similarity)
        attention_scores = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Sum attention scores across rows to get weights
        attention_weights = attention_scores.sum(dim=1)
        
        # Apply softmax to get final weights
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # If top_k is specified, keep only the top k weights
        if self.top_k is not None and self.top_k > 0 and self.top_k < publication_embeddings.shape[0]:
            # Get top k values and indices
            top_k_values, top_k_indices = torch.topk(attention_weights, self.top_k)
            
            # Create mask for top k weights
            mask = torch.zeros_like(attention_weights)
            mask.scatter_(0, top_k_indices, 1.0)
            
            # Apply mask and renormalize
            attention_weights = attention_weights * mask
            attention_weights = attention_weights / attention_weights.sum()
        
        # Compute weighted sum of publication embeddings
        reviewer_embedding = torch.mm(attention_weights.unsqueeze(0), publication_embeddings).squeeze(0)
        
        return reviewer_embedding
    
    def compute_embeddings(self, publication_embeddings_matrix, reviewer_to_pub_ids, pub_id_to_index):
        """
        Compute embeddings for multiple reviewers using self-attention clustering.
        
        Args:
            publication_embeddings_matrix: Matrix of shape (n_total_publications, embedding_dim)
                                          containing embeddings for all publications
            reviewer_to_pub_ids: Dictionary mapping reviewer IDs to lists of their publication IDs
            pub_id_to_index: Dictionary mapping publication IDs to row indices in the embedding matrix
        
        Returns:
            Dictionary mapping reviewer IDs to their computed embeddings
        """
        reviewer_embeddings = {}
        publication_embeddings_matrix = publication_embeddings_matrix.to(self.device)
        
        # Process reviewers in batches to avoid memory issues
        reviewer_ids = list(reviewer_to_pub_ids.keys())
        
        for batch_start in tqdm(range(0, len(reviewer_ids), self.batch_size), desc="Computing reviewer embeddings", total=int(len(reviewer_ids)/self.batch_size), unit="batches"):
            batch_reviewer_ids = reviewer_ids[batch_start:batch_start + self.batch_size]
            
            for reviewer_id in batch_reviewer_ids:
                # Get publication IDs for this reviewer
                pub_ids = reviewer_to_pub_ids[reviewer_id]
                
                # Skip reviewers with no publications
                if not pub_ids:
                    continue
                
                # Get indices of publications in the embedding matrix
                try:
                    indices = [pub_id_to_index[pub_id] for pub_id in pub_ids if pub_id in pub_id_to_index]
                except KeyError as e:
                    print(f"Warning: Publication ID {e} not found in pub_id_to_index mapping. Skipping.")
                    continue
                
                # Skip reviewers with no valid publications
                if not indices:
                    continue
                
                # Extract embeddings for this reviewer's publications
                reviewer_pub_embeddings = publication_embeddings_matrix[indices]
                
                # Compute reviewer embedding using self-attention
                reviewer_embedding = self._compute_reviewer_embedding(reviewer_pub_embeddings)
                
                # Store the computed embedding
                reviewer_embeddings[reviewer_id] = reviewer_embedding
        
        return reviewer_embeddings
    
    def compute_scores(self, publication_embeddings_matrix, submission_embeddings_matrix, 
                       reviewer_to_pub_ids, pub_id_to_index):
        """
        Compute affinity scores between reviewers and submissions.
        
        Args:
            publication_embeddings_matrix: Matrix of shape (n_total_publications, embedding_dim)
                                          containing embeddings for all publications
            submission_embeddings_matrix: Matrix of shape (n_submissions, embedding_dim)
                                         containing embeddings for all submissions
            reviewer_to_pub_ids: Dictionary mapping reviewer IDs to lists of their publication IDs
            pub_id_to_index: Dictionary mapping publication IDs to row indices in the embedding matrix
        
        Returns:
            Dictionary mapping (reviewer_id, submission_id) pairs to affinity scores
        """
        # First compute reviewer embeddings
        reviewer_embeddings = self.compute_embeddings(
            publication_embeddings_matrix, reviewer_to_pub_ids, pub_id_to_index
        )
        
        # Convert submission embeddings to torch tensor
        submission_embeddings_matrix = torch.tensor(submission_embeddings_matrix, device=self.device)
        
        # Normalize submission embeddings for cosine similarity
        normalized_submissions = F.normalize(submission_embeddings_matrix, p=2, dim=1)
        
        scores = {}
        
        # Process reviewers in batches
        reviewer_ids = list(reviewer_embeddings.keys())
        submission_ids = list(range(submission_embeddings_matrix.shape[0]))
        
        for batch_start in range(0, len(reviewer_ids), self.batch_size):
            batch_reviewer_ids = reviewer_ids[batch_start:batch_start + self.batch_size]
            
            # Create batch of reviewer embeddings
            batch_embeddings = [torch.tensor(reviewer_embeddings[r_id], device=self.device) 
                               for r_id in batch_reviewer_ids]
            
            if not batch_embeddings:
                continue
                
            batch_embeddings_tensor = torch.stack(batch_embeddings)
            
            # Normalize for cosine similarity
            normalized_reviewers = F.normalize(batch_embeddings_tensor, p=2, dim=1)
            
            # Compute cosine similarity between reviewers and submissions
            # Shape: (batch_size, n_submissions)
            similarity_scores = torch.mm(normalized_reviewers, normalized_submissions.t())
            
            # Convert to numpy for storage
            similarity_scores = similarity_scores.cpu().numpy()
            
            # Store scores
            for i, reviewer_id in enumerate(batch_reviewer_ids):
                for j, submission_id in enumerate(submission_ids):
                    scores[(reviewer_id, submission_id)] = float(similarity_scores[i, j])
        
        return scores