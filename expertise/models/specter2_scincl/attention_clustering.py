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
    
    def __init__(self, top_k=None, batch_size=32, device=None, cross_attention=False):
        """
        Initialize the self-attention clustering predictor.
        
        Args:
            top_k: Number of top attention weights to keep (None means keep all)
            batch_size: Batch size for processing reviewers in parallel
            device: Device to use for computation (None for automatic detection)
            cross_attention: Whether to include cross-attention between submissions and reviewers
        """
        self.top_k = top_k
        self.batch_size = batch_size
        self.cross_attention = cross_attention
        
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

    def _apply_top_k_and_normalize(self, attention_weights, mask=None, k=None):
        """
        Helper function to apply top-k filtering and normalize weights.
        
        Args:
            attention_weights: Tensor of shape (batch_size, seq_len)
            mask: Optional binary mask of shape (batch_size, seq_len) with 1s for real values, 0s for padding
            k: Number of top weights to keep (uses self.top_k if None)
        
        Returns:
            Filtered and normalized attention weights of shape (batch_size, seq_len)
        """
        if mask is not None:
            # Apply mask to zero out padding
            # Shape: (batch_size, seq_len)
            attention_weights = attention_weights * mask
        
        # Get top-k value (default to self.top_k)
        top_k = k if k is not None else self.top_k
        
        # Apply top-k if specified
        if top_k is not None and top_k > 0:
            batch_size, seq_len = attention_weights.shape
            
            # Vectorized approach for top-k selection
            if mask is not None:
                # Compute effective k for each row based on available valid elements
                # Shape: (batch_size, 1)
                valid_counts = mask.sum(dim=1, keepdim=True).to(torch.int32)
                # Ensure k doesn't exceed valid count for each row
                # Shape: (batch_size, 1)
                effective_k = torch.min(torch.tensor(top_k, device=self.device).expand_as(valid_counts), valid_counts)
                
                # Create a large negative value mask for padding
                # Shape: (batch_size, seq_len)
                neg_inf_mask = (mask == 0) * -1e9
                
                # Apply the negative mask to attention weights
                # Shape: (batch_size, seq_len)
                masked_weights = attention_weights + neg_inf_mask
                
                # Get top-k values and indices - will correctly ignore padded positions
                # due to large negative values
                # Shape: values and indices both (batch_size, top_k)
                top_values, top_indices = torch.topk(masked_weights, min(top_k, seq_len), dim=1)
                
                # Create new mask for top-k weights
                # Initialize with zeros: Shape (batch_size, seq_len)
                top_k_mask = torch.zeros_like(attention_weights)
                
                # Use scatter to set top-k positions to 1
                # - batch_indices: Shape (batch_size, top_k) containing row indices
                # - top_indices: Shape (batch_size, top_k) containing column indices where values should be 1
                batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, top_indices.size(1))
                top_k_mask[batch_indices, top_indices] = 1.0
            else:
                # Simple case - same k for all rows
                # Shape: values and indices both (batch_size, top_k)
                _, top_indices = torch.topk(attention_weights, min(top_k, seq_len), dim=1)
                
                # Create mask for top-k weights
                # Shape: (batch_size, seq_len)
                top_k_mask = torch.zeros_like(attention_weights)
                
                # Use scatter to set top-k positions to 1
                batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, top_indices.size(1))
                top_k_mask[batch_indices, top_indices] = 1.0
            
            # Apply mask to keep only top-k weights
            # Shape: (batch_size, seq_len)
            attention_weights = attention_weights * top_k_mask
        
        # Normalize to sum to 1 along dim=1
        # Shape: (batch_size, 1)
        sum_weights = attention_weights.sum(dim=1, keepdim=True)
        # Handle potential division by zero by adding small epsilon where sum is zero
        # Shape: (batch_size, seq_len)
        normalized_weights = attention_weights / (sum_weights + (sum_weights == 0).float() * 1e-10)
        
        return normalized_weights
    
    def _compute_reviewer_embeddings_batch(self, publication_embeddings_batch, attention_mask, submission_embeddings=None):
        """
        Compute reviewer embeddings for a batch of reviewers using vectorized self-attention.
        If cross_attention=True, combines self-attention with cross-attention in a product-of-experts way.
        
        Args:
            publication_embeddings_batch: Tensor of shape (batch_size, max_pubs, embedding_dim)
                                         containing padded embeddings for all reviewers
            attention_mask: Tensor of shape (batch_size, max_pubs) with 1s for real publications
                          and 0s for padding
            submission_embeddings: Tensor of shape (n_submissions, embedding_dim) containing
                                   the embeddings of all submissions, if cross_attention is True
        
        Returns:
            If cross_attention=False:
                Tensor of shape (batch_size, embedding_dim) containing reviewer embeddings
            If cross_attention=True:
                Tensor of shape (batch_size, n_submissions) containing affinity scores
        """
        batch_size, max_pubs, embedding_dim = publication_embeddings_batch.shape
        
        # Handle single-publication case
        if max_pubs == 1:
            # For single publication reviewers, the embedding is just the publication
            # Shape: (batch_size, embedding_dim)
            reviewer_embeddings = publication_embeddings_batch.squeeze(1)
            
            # If cross_attention, compute scores directly
            if self.cross_attention and submission_embeddings is not None:
                # Normalize embeddings for cosine similarity
                # Shape: (batch_size, embedding_dim)
                normalized_reviewer_embs = F.normalize(reviewer_embeddings, p=2, dim=1)
                # Shape: (n_submissions, embedding_dim)
                normalized_submissions = F.normalize(submission_embeddings, p=2, dim=1)
                
                # Compute cosine similarity
                # Shape: (batch_size, n_submissions)
                scores = torch.mm(normalized_reviewer_embs, normalized_submissions.t())
                return scores
            
            return reviewer_embeddings
        
        # Normalize reviewer publication embeddings for cosine similarity
        # Shape: (batch_size, max_pubs, embedding_dim)
        normalized_pub_embeddings = F.normalize(publication_embeddings_batch, p=2, dim=2)
        
        # COMPUTE SELF-ATTENTION SCORES FOR ALL REVIEWERS
        # This is used in both branches
        
        # Compute attention scores using batch matrix multiplication
        # Shape: (batch_size, max_pubs, max_pubs)
        self_attention_scores = torch.bmm(normalized_pub_embeddings, normalized_pub_embeddings.transpose(1, 2))
        
        # Apply mask to attention scores to ignore padding
        # Create a 2D mask for each reviewer by broadcasting
        # Shape: (batch_size, max_pubs, max_pubs)
        mask_2d = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
        self_attention_scores = self_attention_scores * mask_2d
        
        # Sum attention scores across rows to get weights
        # Shape: (batch_size, max_pubs)
        self_attention_weights = self_attention_scores.sum(dim=2)
        
        # Replace NaN/Inf values with zeros
        # Shape: (batch_size, max_pubs)
        self_attention_weights = torch.where(
            torch.isfinite(self_attention_weights),
            self_attention_weights,
            torch.zeros_like(self_attention_weights)
        )
        
        # Apply mask again to zero out padding
        # Shape: (batch_size, max_pubs)
        self_attention_weights = self_attention_weights * attention_mask
        
        # Handle zero weights with uniform distribution
        # Shape: (batch_size, 1) - boolean tensor indicating which rows have all zeros
        zero_weight_rows = (self_attention_weights.sum(dim=1, keepdim=True) == 0)
        
        if zero_weight_rows.any():
            # For rows with zero weights, set uniform weights for real publications
            # Shape: (batch_size, max_pubs)
            uniform_weights = attention_mask.float() / (attention_mask.sum(dim=1, keepdim=True) + 1e-10)
            
            # Apply uniform weights where needed
            # Shape: (batch_size, max_pubs)
            self_attention_weights = torch.where(
                zero_weight_rows.expand_as(self_attention_weights),
                uniform_weights,
                self_attention_weights
            )
        
        # DIRECT REVIEWER-SUBMISSION SCORING WITH CROSS-ATTENTION
        if self.cross_attention and submission_embeddings is not None:
            # Normalize submission embeddings
            # Shape: (n_submissions, embedding_dim)
            normalized_submissions = F.normalize(submission_embeddings, p=2, dim=1)
            n_submissions = submission_embeddings.shape[0]
            
            # Initialize scores matrix
            # Shape: (batch_size, n_submissions)
            reviewer_submission_scores = torch.zeros((batch_size, n_submissions), device=self.device)
            
            # Process submissions in batches to manage memory
            submission_batch_size = min(100, n_submissions)
            
            for sub_batch_start in range(0, n_submissions, submission_batch_size):
                sub_batch_end = min(sub_batch_start + submission_batch_size, n_submissions)
                # Shape: (sub_batch_size, embedding_dim)
                sub_batch_submissions = normalized_submissions[sub_batch_start:sub_batch_end]
                sub_batch_size = sub_batch_submissions.shape[0]
                
                # VECTORIZED CROSS-ATTENTION COMPUTATION

                # Compute cosine similarity between each reviewer's publications and each submission
                # Shape: (batch_size, max_pubs, sub_batch_size)
                cross_attention_scores = torch.bmm(
                    normalized_pub_embeddings, 
                    sub_batch_submissions.t().unsqueeze(0).expand(batch_size, -1, -1)
                )
                
                # Reshape for more efficient vectorized operations
                # Transpose to get (batch_size, sub_batch_size, max_pubs)
                cross_scores_transposed = cross_attention_scores.transpose(1, 2)
                
                # Expand self_attention_weights for broadcasting
                # Shape: (batch_size, 1, max_pubs)
                self_attn_expanded = self_attention_weights.unsqueeze(1)
                
                # Expand attention_mask for broadcasting
                # Shape: (batch_size, 1, max_pubs)
                mask_expanded = attention_mask.unsqueeze(1)
                
                # Apply mask to cross-attention scores
                # Shape: (batch_size, sub_batch_size, max_pubs)
                cross_scores_masked = cross_scores_transposed * mask_expanded
                
                # PRODUCT OF EXPERTS - Combine self-attention and cross-attention
                # Multiply element-wise: (batch_size, sub_batch_size, max_pubs)
                combined_weights = cross_scores_masked * self_attn_expanded
                
                # Replace NaN/Inf values
                # Shape: (batch_size, sub_batch_size, max_pubs)
                combined_weights = torch.where(
                    torch.isfinite(combined_weights),
                    combined_weights,
                    torch.zeros_like(combined_weights)
                )
                
                # Apply softmax along publication dimension
                # First mask padding with large negative values
                # Shape: (batch_size, sub_batch_size, max_pubs)
                combined_weights = combined_weights.masked_fill(mask_expanded == 0, -1e9)
                attention_weights = F.softmax(combined_weights, dim=2)
                
                # Apply top-k filtering to each (reviewer, submission) pair
                # Reshape for top-k operation
                # Shape: (batch_size * sub_batch_size, max_pubs)
                flat_weights = attention_weights.reshape(-1, max_pubs)
                flat_mask = attention_mask.unsqueeze(1).expand(-1, sub_batch_size, -1).reshape(-1, max_pubs)
                
                # Apply top-k filtering
                # Shape: (batch_size * sub_batch_size, max_pubs)
                filtered_weights = self._apply_top_k_and_normalize(flat_weights, flat_mask)
                
                # Reshape back
                # Shape: (batch_size, sub_batch_size, max_pubs)
                filtered_weights = filtered_weights.reshape(batch_size, sub_batch_size, max_pubs)
                
                # Compute final scores using the filtered weights
                # Shape: (batch_size, sub_batch_size, max_pubs) * (batch_size, sub_batch_size, max_pubs)
                # -> Sum over max_pubs dimension to get (batch_size, sub_batch_size)
                weighted_scores = (cross_scores_transposed * filtered_weights).sum(dim=2)
                
                # Store in the scores matrix
                # Shape: (batch_size, sub_batch_size)
                reviewer_submission_scores[:, sub_batch_start:sub_batch_end] = weighted_scores
            
            return reviewer_submission_scores
        
        # STANDARD SELF-ATTENTION FOR REVIEWER EMBEDDINGS
        else:
            # Apply softmax to get normalized weights from self-attention
            # Shape: (batch_size, max_pubs)
            self_attention_weights = self_attention_weights.masked_fill(attention_mask == 0, -1e9)
            attention_weights = F.softmax(self_attention_weights, dim=1)
            
            # Apply top-k filtering
            # Shape: (batch_size, max_pubs)
            attention_weights = self._apply_top_k_and_normalize(
                attention_weights, mask=attention_mask
            )
            
            # Compute weighted sum of publication embeddings
            # Shape: (batch_size, 1, max_pubs) × (batch_size, max_pubs, embedding_dim)
            # -> (batch_size, 1, embedding_dim) -> (batch_size, embedding_dim)
            reviewer_embeddings = torch.bmm(
                attention_weights.unsqueeze(1), 
                publication_embeddings_batch
            ).squeeze(1)
            
            return reviewer_embeddings
    
    def compute_embeddings(
        self, 
        publication_embeddings_matrix,
        reviewer_to_pub_ids,
        pub_id_to_index,
        submission_embeddings_matrix=None,
        submission_ids=None
    ):
        """
        Compute embeddings for multiple reviewers using vectorized self-attention clustering.
        
        Args:
            publication_embeddings_matrix: Matrix of shape (n_total_publications, embedding_dim)
                                          containing embeddings for all publications
            reviewer_to_pub_ids: Dictionary mapping reviewer IDs to lists of their publication IDs
            pub_id_to_index: Dictionary mapping publication IDs to row indices in the embedding matrix
            submission_embeddings_matrix: Matrix of shape (n_submissions, embedding_dim) containing
                                         embeddings for all submissions, if cross_attention is True
            submission_ids: List of submission IDs, if cross_attention is True
        Returns:
            Dictionary mapping reviewer IDs to their computed embeddings
            OR
            Nested dict of reviewer_id -> submission_id -> score directly if cross_attention is True
        """
        reviewer_embeddings, all_scores = {}, {}
        publication_embeddings_matrix = publication_embeddings_matrix.to(self.device)
        submission_embeddings_matrix = submission_embeddings_matrix.to(self.device) if submission_embeddings_matrix is not None else None
        
        # Process reviewers in batches to avoid memory issues
        reviewer_ids = list(reviewer_to_pub_ids.keys())
        embedding_dim = publication_embeddings_matrix.shape[1]
        
        # Pre-filter valid reviewers and prepare data (this can't be easily vectorized)
        valid_reviewers = []
        publication_indices_list = []
        
        for reviewer_id in reviewer_ids:
            pub_ids = reviewer_to_pub_ids[reviewer_id]
            if not pub_ids:
                continue
                
            try:
                indices = [pub_id_to_index[pub_id] for pub_id in pub_ids if pub_id in pub_id_to_index]
                if not indices:
                    continue
                    
                valid_reviewers.append(reviewer_id)
                publication_indices_list.append(indices)
            except KeyError as e:
                print(f"Warning: Publication ID {e} not found in pub_id_to_index mapping. Skipping reviewer {reviewer_id}.")
        
        # Process in batches
        for batch_start in tqdm(range(0, len(valid_reviewers), self.batch_size), 
                               desc="Computing reviewer embeddings", 
                               total=(len(valid_reviewers) + self.batch_size - 1) // self.batch_size,
                               unit="batches"):
            batch_end = min(batch_start + self.batch_size, len(valid_reviewers))
            batch_reviewer_ids = valid_reviewers[batch_start:batch_end]
            batch_publication_indices = publication_indices_list[batch_start:batch_end]
            
            # Find maximum number of publications for padding
            max_pubs = max(len(indices) for indices in batch_publication_indices)
            
            # Create padded embeddings and attention mask
            batch_size = len(batch_reviewer_ids)
            padded_embeddings = torch.zeros((batch_size, max_pubs, embedding_dim), device=self.device)
            attention_mask = torch.zeros((batch_size, max_pubs), device=self.device)
            
            # Fill in the embeddings and mask
            for i, indices in enumerate(batch_publication_indices):
                n_pubs = len(indices)
                padded_embeddings[i, :n_pubs] = publication_embeddings_matrix[indices]
                attention_mask[i, :n_pubs] = 1.0
            
            # Compute embeddings for the batch
            batch_ret_val = self._compute_reviewer_embeddings_batch(
                padded_embeddings,
                attention_mask,
                submission_embeddings=submission_embeddings_matrix
            )

            # Store embeddings or scores efficiently
            if self.cross_attention and submission_ids is not None:
                # Vectorized storage of scores
                for i, reviewer_id in enumerate(batch_reviewer_ids):
                    all_scores[reviewer_id] = {
                        sub_id: float(batch_ret_val[i, j].item())
                        for j, sub_id in enumerate(submission_ids)
                    }
            else:
                # Vectorized storage of embeddings
                for i, reviewer_id in enumerate(batch_reviewer_ids):
                    reviewer_embeddings[reviewer_id] = batch_ret_val[i].cpu()
        
        if self.cross_attention:
            return all_scores
        else:
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
        # If using cross-attention, compute scores with direct approach
        if self.cross_attention:
            submission_ids = list(range(submission_embeddings_matrix.shape[0]))
            
            # Get nested dictionary of reviewer -> submission -> score
            reviewer_scores = self.compute_embeddings(
                publication_embeddings_matrix, 
                reviewer_to_pub_ids, 
                pub_id_to_index,
                submission_embeddings_matrix,
                submission_ids
            )
            
            # Convert to flat dictionary for consistent return format
            scores = {}
            for reviewer_id, sub_scores in reviewer_scores.items():
                for submission_id, score in sub_scores.items():
                    scores[(reviewer_id, submission_id)] = score
            
            return scores
        
        # Otherwise, use traditional approach: compute embeddings then score
        else:
            # First compute reviewer embeddings
            reviewer_embeddings = self.compute_embeddings(
                publication_embeddings_matrix, reviewer_to_pub_ids, pub_id_to_index
            )
            
            # Convert submission embeddings to torch tensor
            submission_embeddings_matrix = torch.tensor(submission_embeddings_matrix, device=self.device)
            
            # Normalize submission embeddings for cosine similarity
            # Shape: (n_submissions, embedding_dim)
            normalized_submissions = F.normalize(submission_embeddings_matrix, p=2, dim=1)
            
            scores = {}
            reviewer_ids = list(reviewer_embeddings.keys())
            submission_ids = list(range(submission_embeddings_matrix.shape[0]))
            
            # Process reviewers in batches
            for batch_start in range(0, len(reviewer_ids), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(reviewer_ids))
                batch_reviewer_ids = reviewer_ids[batch_start:batch_end]
                
                # Create batch of reviewer embeddings - vectorized
                # Shape: (batch_size, embedding_dim)
                batch_embeddings = torch.stack([
                    torch.tensor(reviewer_embeddings[r_id], device=self.device) 
                    for r_id in batch_reviewer_ids
                ])
                
                # Normalize for cosine similarity
                # Shape: (batch_size, embedding_dim)
                normalized_reviewers = F.normalize(batch_embeddings, p=2, dim=1)
                
                # Compute cosine similarity between reviewers and submissions
                # Shape: (batch_size, n_submissions)
                similarity_scores = torch.mm(normalized_reviewers, normalized_submissions.t())
                
                # Vectorized conversion to dictionary
                for i, reviewer_id in enumerate(batch_reviewer_ids):
                    for j, submission_id in enumerate(submission_ids):
                        scores[(reviewer_id, submission_id)] = float(similarity_scores[i, j].item())
            
            return scores