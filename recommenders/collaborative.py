"""
Collaborative Filtering Recommender

This module implements collaborative filtering for course recommendations.
It uses two approaches:
1. User-User Collaborative Filtering: Find similar users and recommend their courses
2. Item-Item Collaborative Filtering: Find similar courses based on interaction patterns

WHY COLLABORATIVE FILTERING:
- Captures "wisdom of the crowd" - professionals who learned X then learned Y
- Discovers non-obvious course relationships
- Works well when we have interaction data

Author: RecommenderSystem
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from models.data_models import Course, User, Interaction, Recommendation
from config import CF_MIN_INTERACTIONS, CF_NUM_NEIGHBORS, CF_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering Recommender using cosine similarity.
    
    Supports both user-based and item-based collaborative filtering.
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
        
        # Build interaction matrices
        self._build_matrices()
    
    def _build_matrices(self):
        """Build user-course and course-user matrices."""
        # User -> Course -> Score
        self.user_course_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        # Course -> User -> Score  
        self.course_user_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        # Track completed courses per user
        self.user_completed: Dict[str, set] = defaultdict(set)
        
        for interaction in self.interactions:
            # Calculate score based on interaction type
            if interaction.rating is not None:
                score = interaction.rating / 5.0
            elif interaction.completed:
                score = 0.8
            else:
                score = 0.5
            
            self.user_course_matrix[interaction.user_id][interaction.course_id] = score
            self.course_user_matrix[interaction.course_id][interaction.user_id] = score
            
            if interaction.completed:
                self.user_completed[interaction.user_id].add(interaction.course_id)
        
        logger.info(f"Built matrices: {len(self.user_course_matrix)} users, {len(self.course_user_matrix)} courses")
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two sparse vectors.
        
        Args:
            vec1: First vector as {key: value} dict
            vec2: Second vector as {key: value} dict
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Find common keys
        common_keys = set(vec1.keys()) & set(vec2.keys())
        
        if not common_keys:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
        mag1 = np.sqrt(sum(v**2 for v in vec1.values()))
        mag2 = np.sqrt(sum(v**2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _find_similar_users(
        self, 
        user_id: str, 
        top_n: int = CF_NUM_NEIGHBORS
    ) -> List[Tuple[str, float]]:
        """
        Find users most similar to the given user based on course interactions.
        
        Args:
            user_id: Target user ID
            top_n: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if user_id not in self.user_course_matrix:
            return []
        
        target_vec = self.user_course_matrix[user_id]
        similarities = []
        
        for other_id, other_vec in self.user_course_matrix.items():
            if other_id == user_id:
                continue
            
            sim = self._cosine_similarity(target_vec, other_vec)
            if sim >= CF_SIMILARITY_THRESHOLD:
                similarities.append((other_id, sim))
        
        # Sort by similarity (descending) and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def _find_similar_courses(
        self, 
        course_id: str, 
        top_n: int = CF_NUM_NEIGHBORS
    ) -> List[Tuple[str, float]]:
        """
        Find courses similar to the given course based on user interaction patterns.
        
        Args:
            course_id: Target course ID
            top_n: Number of similar courses to return
            
        Returns:
            List of (course_id, similarity_score) tuples
        """
        if course_id not in self.course_user_matrix:
            return []
        
        target_vec = self.course_user_matrix[course_id]
        similarities = []
        
        for other_id, other_vec in self.course_user_matrix.items():
            if other_id == course_id:
                continue
            
            sim = self._cosine_similarity(target_vec, other_vec)
            if sim >= CF_SIMILARITY_THRESHOLD:
                similarities.append((other_id, sim))
        
        # Sort by similarity (descending) and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def recommend_user_based(
        self, 
        user_id: str, 
        top_k: int = 10,
        exclude_completed: bool = True
    ) -> List[Tuple[str, float, str]]:
        """
        Generate recommendations using user-based collaborative filtering.
        
        Logic: Find users with similar learning patterns and recommend courses
        they completed that the target user hasn't taken.
        
        Args:
            user_id: Target user ID
            top_k: Number of recommendations to return
            exclude_completed: Whether to exclude courses user already completed
            
        Returns:
            List of (course_id, score, reason) tuples
        """
        # Check if user has enough interactions
        user_interactions = len(self.user_course_matrix.get(user_id, {}))
        if user_interactions < CF_MIN_INTERACTIONS:
            logger.info(f"User {user_id} has {user_interactions} interactions (< {CF_MIN_INTERACTIONS})")
            return []
        
        # Find similar users
        similar_users = self._find_similar_users(user_id)
        if not similar_users:
            logger.info(f"No similar users found for {user_id}")
            return []
        
        # Get courses completed by similar users
        user_completed = self.user_completed.get(user_id, set())
        course_scores: Dict[str, Tuple[float, str]] = {}
        
        for sim_user_id, similarity in similar_users:
            sim_user_completed = self.user_completed.get(sim_user_id, set())
            
            for course_id in sim_user_completed:
                # Skip if user already completed this course
                if exclude_completed and course_id in user_completed:
                    continue
                
                # Skip if course not in catalog
                if course_id not in self.courses:
                    continue
                
                # Weight by similarity and rating
                rating = self.user_course_matrix[sim_user_id].get(course_id, 0.8)
                weighted_score = similarity * rating
                
                if course_id not in course_scores or weighted_score > course_scores[course_id][0]:
                    reason = f"Completed by similar user with {similarity:.0%} match"
                    course_scores[course_id] = (weighted_score, reason)
        
        # Sort and return top K
        sorted_courses = sorted(course_scores.items(), key=lambda x: x[1][0], reverse=True)
        return [(cid, score, reason) for cid, (score, reason) in sorted_courses[:top_k]]
    
    def recommend_item_based(
        self, 
        user_id: str, 
        top_k: int = 10,
        exclude_completed: bool = True
    ) -> List[Tuple[str, float, str]]:
        """
        Generate recommendations using item-based collaborative filtering.
        
        Logic: Based on courses the user has completed, find similar courses.
        
        Args:
            user_id: Target user ID
            top_k: Number of recommendations to return
            exclude_completed: Whether to exclude courses user already completed
            
        Returns:
            List of (course_id, score, reason) tuples
        """
        user_completed = self.user_completed.get(user_id, set())
        
        if not user_completed:
            logger.info(f"User {user_id} has no completed courses for item-based CF")
            return []
        
        course_scores: Dict[str, Tuple[float, str]] = {}
        
        for completed_course_id in user_completed:
            similar_courses = self._find_similar_courses(completed_course_id)
            
            for similar_id, similarity in similar_courses:
                # Skip if user already completed this course
                if exclude_completed and similar_id in user_completed:
                    continue
                
                # Skip if course not in catalog
                if similar_id not in self.courses:
                    continue
                
                if similar_id not in course_scores or similarity > course_scores[similar_id][0]:
                    source_title = self.courses[completed_course_id].title[:30]
                    reason = f"Similar to '{source_title}...' ({similarity:.0%} match)"
                    course_scores[similar_id] = (similarity, reason)
        
        # Sort and return top K
        sorted_courses = sorted(course_scores.items(), key=lambda x: x[1][0], reverse=True)
        return [(cid, score, reason) for cid, (score, reason) in sorted_courses[:top_k]]
    
    def recommend(
        self, 
        user_id: str, 
        top_k: int = 10,
        method: str = "combined"
    ) -> List[Tuple[str, float, str]]:
        """
        Generate collaborative filtering recommendations.
        
        Args:
            user_id: Target user ID
            top_k: Number of recommendations to return
            method: "user_based", "item_based", or "combined"
            
        Returns:
            List of (course_id, score, reason) tuples
        """
        if method == "user_based":
            return self.recommend_user_based(user_id, top_k)
        elif method == "item_based":
            return self.recommend_item_based(user_id, top_k)
        else:
            # Combine both methods
            user_recs = self.recommend_user_based(user_id, top_k)
            item_recs = self.recommend_item_based(user_id, top_k)
            
            # Merge and deduplicate
            combined: Dict[str, Tuple[float, str]] = {}
            
            for course_id, score, reason in user_recs:
                combined[course_id] = (score, reason)
            
            for course_id, score, reason in item_recs:
                if course_id in combined:
                    # Take max score, combine reasons
                    existing_score, existing_reason = combined[course_id]
                    if score > existing_score:
                        combined[course_id] = (score, reason)
                else:
                    combined[course_id] = (score, reason)
            
            # Sort and return top K
            sorted_courses = sorted(combined.items(), key=lambda x: x[1][0], reverse=True)
            return [(cid, score, reason) for cid, (score, reason) in sorted_courses[:top_k]]
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics about user's interaction data for debugging."""
        return {
            "total_interactions": len(self.user_course_matrix.get(user_id, {})),
            "completed_courses": len(self.user_completed.get(user_id, set())),
            "similar_users_count": len(self._find_similar_users(user_id)),
            "has_enough_data": len(self.user_course_matrix.get(user_id, {})) >= CF_MIN_INTERACTIONS
        }
