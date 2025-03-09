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
    
    def _compute_reviewer_embeddings_batch(self, publication_embeddings_batch, attention_mask):
        """
        Compute reviewer embeddings for a batch of reviewers using vectorized self-attention.
        
        Args:
            publication_embeddings_batch: Tensor of shape (batch_size, max_pubs, embedding_dim)
                                         containing padded embeddings for all reviewers
            attention_mask: Tensor of shape (batch_size, max_pubs) with 1s for real publications
                          and 0s for padding
        
        Returns:
            Tensor of shape (batch_size, embedding_dim) containing reviewer embeddings
        """
        batch_size, max_pubs, embedding_dim = publication_embeddings_batch.shape
        
        # Handle single-publication case
        if max_pubs == 1:
            return publication_embeddings_batch.squeeze(1)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = F.normalize(publication_embeddings_batch, p=2, dim=2)
        
        # Compute attention scores (shape: batch_size, max_pubs, max_pubs)
        # bmm performs batch matrix multiplication
        attention_scores = torch.bmm(normalized_embeddings, normalized_embeddings.transpose(1, 2))
        
        # Apply mask to attention scores to ignore padding
        # Create a 2D mask for each reviewer (expanded for broadcasting)
        mask_2d = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
        
        # Apply the mask to zero out attention scores for padding
        attention_scores = attention_scores * mask_2d
        
        # Sum attention scores across rows to get weights (shape: batch_size, max_pubs)
        attention_weights = attention_scores.sum(dim=2)
        
        # Replace NaN/Inf values that might occur with zeros 
        attention_weights = torch.where(
            torch.isfinite(attention_weights),
            attention_weights,
            torch.zeros_like(attention_weights)
        )
        
        # Apply mask again to ensure only real publications have weights
        attention_weights = attention_weights * attention_mask
        
        # Handle zero weights (if any) by replacing with uniform distribution over real publications
        zero_weight_rows = (attention_weights.sum(dim=1) == 0)
        if zero_weight_rows.any():
            # For rows with zero weights, set uniform weights for real publications
            uniform_weights = attention_mask.float() / (attention_mask.sum(dim=1, keepdim=True) + 1e-10)
            attention_weights = torch.where(
                zero_weight_rows.unsqueeze(1).expand_as(attention_weights),
                uniform_weights,
                attention_weights
            )
        
        # Apply softmax to get final weights (with masking to ignore padding)
        # First add a large negative value to padded positions
        attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply top-k if specified
        if self.top_k is not None and self.top_k > 0:
            # Get top-k values and indices
            top_k = min(self.top_k, max_pubs)
            _, top_indices = torch.topk(attention_weights, top_k, dim=1)
            
            # Create a mask for top-k weights
            top_k_mask = torch.zeros_like(attention_weights)
            
            # Use scatter to set top-k positions to 1
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, top_k).to(self.device)
            top_k_mask[batch_indices, top_indices] = 1.0
            
            # Apply mask and renormalize
            attention_weights = attention_weights * top_k_mask
            # Handle potential division by zero
            sum_weights = attention_weights.sum(dim=1, keepdim=True)
            attention_weights = attention_weights / (sum_weights + (sum_weights == 0).float())
        
        # Compute weighted sum of publication embeddings (shape: batch_size, embedding_dim)
        reviewer_embeddings = torch.bmm(attention_weights.unsqueeze(1), publication_embeddings_batch).squeeze(1)
        
        return reviewer_embeddings
    
    def compute_embeddings(self, publication_embeddings_matrix, reviewer_to_pub_ids, pub_id_to_index):
        """
        Compute embeddings for multiple reviewers using vectorized self-attention clustering.
        
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
        embedding_dim = publication_embeddings_matrix.shape[1]
        
        for batch_start in tqdm(range(0, len(reviewer_ids), self.batch_size), 
                               desc="Computing reviewer embeddings", 
                               total=(len(reviewer_ids) + self.batch_size - 1) // self.batch_size,
                               unit="batches"):
            batch_reviewer_ids = reviewer_ids[batch_start:batch_start + self.batch_size]
            
            # Collect valid reviewers and their publication indices
            valid_reviewers = []
            publication_indices_list = []
            
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
                
                valid_reviewers.append(reviewer_id)
                publication_indices_list.append(indices)
            
            # Skip if no valid reviewers in this batch
            if not valid_reviewers:
                continue
            
            # Find maximum number of publications for padding
            max_pubs = max(len(indices) for indices in publication_indices_list)
            
            # Create padded embeddings and attention mask
            batch_size = len(valid_reviewers)
            padded_embeddings = torch.zeros((batch_size, max_pubs, embedding_dim), device=self.device)
            attention_mask = torch.zeros((batch_size, max_pubs), device=self.device)
            
            # Fill in the embeddings and mask
            for i, indices in enumerate(publication_indices_list):
                n_pubs = len(indices)
                padded_embeddings[i, :n_pubs] = publication_embeddings_matrix[indices]
                attention_mask[i, :n_pubs] = 1.0
            
            # Compute embeddings for the batch
            batch_embeddings = self._compute_reviewer_embeddings_batch(padded_embeddings, attention_mask)
            
            # Store the computed embeddings
            for i, reviewer_id in enumerate(valid_reviewers):
                reviewer_embeddings[reviewer_id] = batch_embeddings[i].cpu()
        
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