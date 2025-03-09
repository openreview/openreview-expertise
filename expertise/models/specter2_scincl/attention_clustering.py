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
            attention_weights: Tensor of attention weights
            mask: Optional binary mask (1 for real values, 0 for padding)
            k: Number of top weights to keep (uses self.top_k if None)
        
        Returns:
            Filtered and normalized attention weights
        """
        if mask is not None:
            # Apply mask to zero out padding
            attention_weights = attention_weights * mask
        
        # Get top-k value (default to self.top_k)
        top_k = k if k is not None else self.top_k
        
        # Apply top-k if specified
        if top_k is not None and top_k > 0:
            if mask is not None:
                # Ensure we don't select more elements than exist in each row
                effective_k = torch.min(
                    torch.tensor(top_k, device=self.device),
                    mask.sum(dim=-1, keepdim=True).to(torch.int32)
                )
                # For each row, we need the correct k value
                k_values = effective_k.squeeze(-1)
                
                # Apply top-k with different k for each row
                batch_size = attention_weights.shape[0]
                row_indices = torch.arange(batch_size, device=self.device)
                
                # Initialize mask for top-k weights
                top_k_mask = torch.zeros_like(attention_weights)
                
                # Apply top-k for each row
                for i in range(batch_size):
                    if k_values[i] > 0:
                        _, top_indices = torch.topk(attention_weights[i], k_values[i])
                        top_k_mask[i].scatter_(0, top_indices, 1.0)
            else:
                # Simple case - same k for all rows
                _, top_indices = torch.topk(attention_weights, min(top_k, attention_weights.shape[-1]), dim=-1)
                
                # Create mask for top-k weights
                batch_indices = torch.arange(attention_weights.shape[0], device=self.device)
                batch_indices = batch_indices.unsqueeze(-1).expand(-1, top_indices.shape[-1])
                
                top_k_mask = torch.zeros_like(attention_weights)
                top_k_mask.scatter_(-1, top_indices, 1.0)
            
            # Apply mask to keep only top-k weights
            attention_weights = attention_weights * top_k_mask
        
        # Normalize to sum to 1
        sum_weights = attention_weights.sum(dim=-1, keepdim=True)
        # Handle potential division by zero
        normalized_weights = attention_weights / (sum_weights + (sum_weights == 0).float())
        
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
            reviewer_embeddings = publication_embeddings_batch.squeeze(1)
            
            # If cross_attention, compute scores directly
            if self.cross_attention and submission_embeddings is not None:
                normalized_reviewer_embs = F.normalize(reviewer_embeddings, p=2, dim=1)
                normalized_submissions = F.normalize(submission_embeddings, p=2, dim=1)
                scores = torch.mm(normalized_reviewer_embs, normalized_submissions.t())
                return scores
            
            return reviewer_embeddings
        
        # Normalize reviewer publication embeddings for cosine similarity
        normalized_pub_embeddings = F.normalize(publication_embeddings_batch, p=2, dim=2)
        
        # COMPUTE SELF-ATTENTION SCORES FOR ALL REVIEWERS
        # This is used in both branches
        
        # Compute attention scores (shape: batch_size, max_pubs, max_pubs)
        self_attention_scores = torch.bmm(normalized_pub_embeddings, normalized_pub_embeddings.transpose(1, 2))
        
        # Apply mask to attention scores to ignore padding
        mask_2d = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
        self_attention_scores = self_attention_scores * mask_2d
        
        # Sum attention scores across rows to get weights (shape: batch_size, max_pubs)
        self_attention_weights = self_attention_scores.sum(dim=2)
        
        # Replace NaN/Inf values with zeros
        self_attention_weights = torch.where(
            torch.isfinite(self_attention_weights),
            self_attention_weights,
            torch.zeros_like(self_attention_weights)
        )
        
        # Apply mask again to zero out padding
        self_attention_weights = self_attention_weights * attention_mask
        
        # Handle zero weights with uniform distribution
        zero_weight_rows = (self_attention_weights.sum(dim=1) == 0)
        if zero_weight_rows.any():
            uniform_weights = attention_mask.float() / (attention_mask.sum(dim=1, keepdim=True) + 1e-10)
            self_attention_weights = torch.where(
                zero_weight_rows.unsqueeze(1).expand_as(self_attention_weights),
                uniform_weights,
                self_attention_weights
            )
        
        # DIRECT REVIEWER-SUBMISSION SCORING WITH CROSS-ATTENTION
        if self.cross_attention and submission_embeddings is not None:
            # Normalize submission embeddings
            normalized_submissions = F.normalize(submission_embeddings, p=2, dim=1)
            n_submissions = submission_embeddings.shape[0]
            
            # Initialize scores matrix (batch_size × n_submissions)
            reviewer_submission_scores = torch.zeros((batch_size, n_submissions), device=self.device)
            
            # Process submissions in batches to manage memory
            submission_batch_size = min(100, n_submissions)
            
            for sub_batch_start in range(0, n_submissions, submission_batch_size):
                sub_batch_end = min(sub_batch_start + submission_batch_size, n_submissions)
                sub_batch_submissions = normalized_submissions[sub_batch_start:sub_batch_end]
                sub_batch_size = sub_batch_submissions.shape[0]
                
                # Compute cosine similarity between each submission and each reviewer's publications
                # Shape: (batch_size, max_pubs, sub_batch_size)
                cross_attention_scores = torch.bmm(
                    normalized_pub_embeddings, 
                    sub_batch_submissions.t().unsqueeze(0).expand(batch_size, -1, -1)
                )
                
                # Iterate through each submission in the batch
                for sub_idx in range(sub_batch_size):
                    # Get cross-attention weights for current submission
                    cross_weights = cross_attention_scores[:, :, sub_idx]  # (batch_size, max_pubs)
                    
                    # Apply mask to zero out padding
                    cross_weights = cross_weights * attention_mask
                    
                    # PRODUCT OF EXPERTS - Combine self-attention and cross-attention
                    # Multiply the weights element-wise
                    combined_weights = self_attention_weights * cross_weights
                    
                    # Replace any NaN/Inf values
                    combined_weights = torch.where(
                        torch.isfinite(combined_weights),
                        combined_weights,
                        torch.zeros_like(combined_weights)
                    )
                    
                    # Apply softmax to normalize the combined weights
                    # First mask out padding by setting large negative value
                    combined_weights = combined_weights.masked_fill(attention_mask == 0, -1e9)
                    attention_weights = F.softmax(combined_weights, dim=1)
                    
                    # Apply top-k filtering if specified
                    attention_weights = self._apply_top_k_and_normalize(
                        attention_weights, mask=attention_mask
                    )
                    
                    # Compute weighted score for current submission
                    # Dot product between attention weights and publication-submission similarities
                    weighted_score = (cross_weights * attention_weights).sum(dim=1)
                    
                    # Store in the scores matrix
                    reviewer_submission_scores[:, sub_batch_start + sub_idx] = weighted_score
            
            return reviewer_submission_scores
        
        # STANDARD SELF-ATTENTION FOR REVIEWER EMBEDDINGS
        else:
            # Apply softmax to get normalized weights from self-attention
            self_attention_weights = self_attention_weights.masked_fill(attention_mask == 0, -1e9)
            attention_weights = F.softmax(self_attention_weights, dim=1)
            
            # Apply top-k filtering
            attention_weights = self._apply_top_k_and_normalize(
                attention_weights, mask=attention_mask
            )
            
            # Compute weighted sum of publication embeddings
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
            batch_ret_val = self._compute_reviewer_embeddings_batch(
                padded_embeddings,
                attention_mask,
                submission_embeddings=submission_embeddings_matrix
            )

            # Store embeddings or scores
            if self.cross_attention:
                # If cross attention, batch_ret_val is a subset of all scores -> copy into all_scores
                for r_idx, reviewer_id in enumerate(valid_reviewers):
                    all_scores[reviewer_id] = {}
                    for s_idx, submission_id in enumerate(submission_ids):
                        all_scores[reviewer_id][submission_id] = batch_ret_val[r_idx, s_idx]
            else:
                # If self attention, batch_ret_val is reviewer embeddings -> copy into reviewer_embeddings
                for i, reviewer_id in enumerate(valid_reviewers):
                    reviewer_embeddings[reviewer_id] = batch_ret_val[i]
        
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