"""
Evaluation Metrics for Recommender System

This module implements standard recommendation metrics to evaluate system quality.
Metrics include:
- Precision@K: Relevant items in top-K
- Recall@K: Coverage of relevant items  
- NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
- Coverage: Percentage of catalog recommended
- Diversity: Intra-list diversity

Author: RecommenderSystem
"""

import numpy as np
from typing import Dict, List, Set
from collections import defaultdict
import logging

from models.data_models import Course, User, Interaction, Recommendation

logger = logging.getLogger(__name__)


def precision_at_k(
    recommended: List[str], 
    relevant: Set[str], 
    k: int
) -> float:
    """
    Calculate Precision@K: proportion of recommended items that are relevant.
    
    Args:
        recommended: List of recommended course IDs (ordered)
        relevant: Set of relevant course IDs
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K score between 0 and 1
    """
    if k <= 0:
        return 0.0
    
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    
    relevant_in_top_k = len(set(top_k) & relevant)
    return relevant_in_top_k / k


def recall_at_k(
    recommended: List[str], 
    relevant: Set[str], 
    k: int
) -> float:
    """
    Calculate Recall@K: proportion of relevant items that are recommended.
    
    Args:
        recommended: List of recommended course IDs (ordered)
        relevant: Set of relevant course IDs
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K score between 0 and 1
    """
    if not relevant:
        return 0.0
    
    top_k = recommended[:k]
    relevant_in_top_k = len(set(top_k) & relevant)
    return relevant_in_top_k / len(relevant)


def dcg_at_k(scores: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    Args:
        scores: List of relevance scores (ordered by recommendation rank)
        k: Number of top items to consider
        
    Returns:
        DCG@K value
    """
    scores = scores[:k]
    if not scores:
        return 0.0
    
    # DCG = sum(rel_i / log2(i+2)) for i in 0..k-1
    gains = [score / np.log2(i + 2) for i, score in enumerate(scores)]
    return sum(gains)


def ndcg_at_k(
    recommended: List[str],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        recommended: List of recommended course IDs (ordered)
        relevance_scores: Dict mapping course_id to relevance score
        k: Number of top recommendations to consider
        
    Returns:
        NDCG@K score between 0 and 1
    """
    # Get relevance scores for recommended items
    rec_scores = [relevance_scores.get(cid, 0.0) for cid in recommended[:k]]
    
    # Calculate DCG
    dcg = dcg_at_k(rec_scores, k)
    
    # Calculate ideal DCG (best possible ordering)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def catalog_coverage(
    all_recommendations: List[List[str]],
    catalog_size: int
) -> float:
    """
    Calculate catalog coverage: percentage of items ever recommended.
    
    Args:
        all_recommendations: List of recommendation lists (one per user)
        catalog_size: Total number of items in catalog
        
    Returns:
        Coverage percentage between 0 and 1
    """
    if catalog_size == 0:
        return 0.0
    
    unique_recommended = set()
    for recs in all_recommendations:
        unique_recommended.update(recs)
    
    return len(unique_recommended) / catalog_size


def intra_list_diversity(
    recommended: List[str],
    courses: Dict[str, Course]
) -> float:
    """
    Calculate intra-list diversity based on category variety.
    
    Args:
        recommended: List of recommended course IDs
        courses: Course catalog
        
    Returns:
        Diversity score between 0 and 1
    """
    if len(recommended) <= 1:
        return 1.0  # Single item is maximally diverse with itself
    
    categories = []
    for cid in recommended:
        if cid in courses:
            categories.append(courses[cid].category)
    
    if not categories:
        return 0.0
    
    # Diversity = unique categories / total items (normalized)
    unique_categories = len(set(categories))
    return unique_categories / len(categories)


class RecommenderEvaluator:
    """
    Evaluator class for comprehensive recommendation system evaluation.
    """
    
    def __init__(
        self,
        courses: Dict[str, Course],
        users: Dict[str, User],
        interactions: List[Interaction]
    ):
        self.courses = courses
        self.users = users
        self.interactions = interactions
        
        # Build ground truth: courses that users completed with high ratings
        self._build_ground_truth()
    
    def _build_ground_truth(self):
        """Build ground truth relevant sets for each user."""
        self.ground_truth: Dict[str, Set[str]] = defaultdict(set)
        self.user_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        for interaction in self.interactions:
            if interaction.completed:
                self.ground_truth[interaction.user_id].add(interaction.course_id)
                
                if interaction.rating:
                    self.user_ratings[interaction.user_id][interaction.course_id] = interaction.rating / 5.0
                else:
                    self.user_ratings[interaction.user_id][interaction.course_id] = 0.8
    
    def evaluate_recommendations(
        self,
        user_id: str,
        recommendations: List[str],
        k_values: List[int] = [5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for a single user.
        
        Args:
            user_id: User ID
            recommendations: List of recommended course IDs
            k_values: Values of K for @K metrics
            
        Returns:
            Dictionary of metric names to values
        """
        relevant = self.ground_truth.get(user_id, set())
        ratings = self.user_ratings.get(user_id, {})
        
        metrics = {}
        
        for k in k_values:
            metrics[f'precision@{k}'] = precision_at_k(recommendations, relevant, k)
            metrics[f'recall@{k}'] = recall_at_k(recommendations, relevant, k)
            metrics[f'ndcg@{k}'] = ndcg_at_k(recommendations, ratings, k)
        
        metrics['diversity'] = intra_list_diversity(recommendations, self.courses)
        
        return metrics
    
    def evaluate_system(
        self,
        recommender,
        user_ids: List[str] = None,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate recommender system across multiple users.
        
        Args:
            recommender: Recommender object with recommend() method
            user_ids: List of user IDs to evaluate (default: all)
            top_k: Number of recommendations per user
            
        Returns:
            Dictionary of aggregated metrics
        """
        if user_ids is None:
            user_ids = list(self.users.keys())
        
        all_metrics = defaultdict(list)
        all_recommendations = []
        
        for user_id in user_ids:
            # Skip users with no ground truth
            if user_id not in self.ground_truth:
                continue
            
            # Get recommendations
            result = recommender.recommend(user_id, top_k)
            rec_ids = [r.course.id for r in result.recommendations]
            all_recommendations.append(rec_ids)
            
            # Evaluate
            user_metrics = self.evaluate_recommendations(user_id, rec_ids)
            for metric_name, value in user_metrics.items():
                all_metrics[metric_name].append(value)
        
        # Aggregate metrics (mean)
        aggregated = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregated[f'mean_{metric_name}'] = np.mean(values)
                aggregated[f'std_{metric_name}'] = np.std(values)
        
        # Add coverage
        aggregated['catalog_coverage'] = catalog_coverage(
            all_recommendations, 
            len(self.courses)
        )
        
        return aggregated
    
    def compare_recommenders(
        self,
        recommenders: Dict[str, object],
        user_ids: List[str] = None,
        top_k: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple recommenders side by side.
        
        Args:
            recommenders: Dictionary of {name: recommender_object}
            user_ids: List of user IDs to evaluate
            top_k: Number of recommendations per user
            
        Returns:
            Dictionary of {recommender_name: metrics}
        """
        results = {}
        
        for name, recommender in recommenders.items():
            logger.info(f"Evaluating {name}...")
            results[name] = self.evaluate_system(recommender, user_ids, top_k)
        
        return results


# Package init
__all__ = [
    'precision_at_k', 'recall_at_k', 'ndcg_at_k',
    'catalog_coverage', 'intra_list_diversity',
    'RecommenderEvaluator'
]
