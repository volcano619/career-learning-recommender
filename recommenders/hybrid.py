"""
Hybrid Recommender System

This module combines multiple recommendation strategies into a unified hybrid system.
It intelligently weights different recommenders based on:
1. Data availability (cold-start handling)
2. User context (new vs. experienced user)
3. Recommendation quality signals

WHY HYBRID:
- Best of all worlds: collaborative + content + knowledge graph
- Graceful degradation when data is sparse
- More diverse and robust recommendations

Author: RecommenderSystem
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from models.data_models import (
    Course, User, Interaction, CareerPath, 
    Recommendation, RecommendationResult, SkillGap
)
from recommenders.collaborative import CollaborativeFilteringRecommender
from recommenders.content_based import ContentBasedRecommender
from recommenders.knowledge_graph import KnowledgeGraphRecommender
from config import (
    HYBRID_WEIGHTS, COLD_START_WEIGHTS, 
    CF_MIN_INTERACTIONS, MIN_SCORE_THRESHOLD,
    DIVERSITY_WEIGHT, MAX_SAME_CATEGORY
)

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid Recommender that combines collaborative, content-based, 
    and knowledge graph approaches.
    """
    
    def __init__(
        self,
        courses: Dict[str, Course],
        users: Dict[str, User],
        interactions: List[Interaction],
        career_paths: Dict[str, CareerPath],
        skills_taxonomy: Dict
    ):
        self.courses = courses
        self.users = users
        self.interactions = interactions
        self.career_paths = career_paths
        self.skills_taxonomy = skills_taxonomy
        
        # Initialize individual recommenders
        self.cf_recommender = CollaborativeFilteringRecommender(
            courses, users, interactions
        )
        self.cb_recommender = ContentBasedRecommender(
            courses, users, career_paths, skills_taxonomy
        )
        self.kg_recommender = KnowledgeGraphRecommender(
            courses, users, career_paths, skills_taxonomy
        )
        
        logger.info("Hybrid recommender initialized with all components")
    
    def _get_weights(self, user_id: str) -> Dict[str, float]:
        """
        Determine weights for each recommender based on user data availability.
        
        Cold-start users get more weight on content-based and knowledge graph.
        Users with history get more weight on collaborative filtering.
        """
        cf_stats = self.cf_recommender.get_user_stats(user_id)
        
        if cf_stats['has_enough_data']:
            # User has enough interaction data
            weights = HYBRID_WEIGHTS.copy()
            
            # Boost CF if user has many similar users
            if cf_stats['similar_users_count'] >= 5:
                weights['collaborative'] += 0.1
                weights['content_based'] -= 0.05
                weights['knowledge_graph'] -= 0.05
        else:
            # Cold-start: rely more on content and knowledge graph
            weights = COLD_START_WEIGHTS.copy()
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def _merge_recommendations(
        self,
        cf_recs: List[Tuple[str, float, str]],
        cb_recs: List[Tuple[str, float, str, List[str]]],
        kg_recs: List[Tuple[str, float, str, int]],
        weights: Dict[str, float],
        top_k: int
    ) -> List[Dict]:
        """
        Merge recommendations from all sources with weighted scoring.
        
        Returns:
            List of merged recommendation dictionaries
        """
        merged: Dict[str, Dict] = {}
        
        # Process collaborative filtering recommendations
        for course_id, score, reason in cf_recs:
            if course_id not in merged:
                merged[course_id] = {
                    'course_id': course_id,
                    'scores': {},
                    'reasons': [],
                    'skills_addressed': [],
                    'sources': []
                }
            merged[course_id]['scores']['collaborative'] = score
            merged[course_id]['reasons'].append(f"[CF] {reason}")
            merged[course_id]['sources'].append('collaborative')
        
        # Process content-based recommendations
        for course_id, score, reason, skills in cb_recs:
            if course_id not in merged:
                merged[course_id] = {
                    'course_id': course_id,
                    'scores': {},
                    'reasons': [],
                    'skills_addressed': [],
                    'sources': []
                }
            merged[course_id]['scores']['content_based'] = score
            merged[course_id]['reasons'].append(f"[CB] {reason}")
            merged[course_id]['skills_addressed'].extend(skills)
            merged[course_id]['sources'].append('content_based')
        
        # Process knowledge graph recommendations
        for course_id, score, reason, seq in kg_recs:
            if course_id not in merged:
                merged[course_id] = {
                    'course_id': course_id,
                    'scores': {},
                    'reasons': [],
                    'skills_addressed': [],
                    'sources': []
                }
            merged[course_id]['scores']['knowledge_graph'] = score
            merged[course_id]['reasons'].append(f"[KG] {reason}")
            merged[course_id]['sources'].append('knowledge_graph')
        
        # Calculate final weighted scores
        for course_id, data in merged.items():
            weighted_score = 0.0
            for source, weight in weights.items():
                if source in data['scores']:
                    weighted_score += weight * data['scores'][source]
            data['final_score'] = weighted_score
            
            # Deduplicate skills_addressed
            data['skills_addressed'] = list(set(data['skills_addressed']))
        
        # Sort by final score
        sorted_recs = sorted(merged.values(), key=lambda x: x['final_score'], reverse=True)
        
        # Apply diversity filter
        return self._apply_diversity(sorted_recs, top_k)
    
    def _apply_diversity(
        self, 
        recommendations: List[Dict], 
        top_k: int
    ) -> List[Dict]:
        """
        Apply diversity filtering to avoid too many courses from same category.
        """
        result = []
        category_counts: Dict[str, int] = defaultdict(int)
        
        for rec in recommendations:
            course = self.courses.get(rec['course_id'])
            if not course:
                continue
            
            category = course.category
            
            # Apply category limit
            if category_counts[category] >= MAX_SAME_CATEGORY:
                # Still add if score is very high
                if rec['final_score'] > 0.8:
                    result.append(rec)
                continue
            
            category_counts[category] += 1
            result.append(rec)
            
            if len(result) >= top_k:
                break
        
        return result[:top_k]
    
    def recommend(
        self, 
        user_id: str, 
        top_k: int = 10
    ) -> RecommendationResult:
        """
        Generate hybrid recommendations for a user.
        
        Args:
            user_id: Target user ID
            top_k: Number of recommendations to return
            
        Returns:
            RecommendationResult with ranked courses and explanations
        """
        if user_id not in self.users:
            logger.warning(f"User {user_id} not found")
            return RecommendationResult(
                user_id=user_id,
                target_role="unknown",
                recommendations=[],
                total_missing_skills=0,
                estimated_learning_hours=0
            )
        
        user = self.users[user_id]
        
        # Get dynamic weights
        weights = self._get_weights(user_id)
        logger.info(f"Using weights for {user_id}: {weights}")
        
        # Get recommendations from each source
        # Request more than needed to have options after diversity filtering
        request_k = top_k * 2
        
        cf_recs = self.cf_recommender.recommend(user_id, request_k)
        cb_recs = self.cb_recommender.recommend(user_id, request_k)
        kg_recs = self.kg_recommender.recommend(user_id, request_k)
        
        logger.info(f"Got {len(cf_recs)} CF, {len(cb_recs)} CB, {len(kg_recs)} KG recommendations")
        
        # Merge recommendations
        merged = self._merge_recommendations(cf_recs, cb_recs, kg_recs, weights, top_k)
        
        # Filter by minimum score
        filtered = [r for r in merged if r['final_score'] >= MIN_SCORE_THRESHOLD]
        
        # Build Recommendation objects
        recommendations = []
        total_hours = 0
        
        for rank, rec_data in enumerate(filtered, 1):
            course = self.courses.get(rec_data['course_id'])
            if not course:
                continue
            
            recommendation = Recommendation(
                course=course,
                score=rec_data['final_score'],
                rank=rank,
                reasons=rec_data['reasons'][:3],  # Top 3 reasons
                skill_gaps_addressed=rec_data['skills_addressed'][:5],
                source='+'.join(set(rec_data['sources']))
            )
            recommendations.append(recommendation)
            total_hours += course.duration_hours
        
        # Get skill gap summary
        skill_gaps = self.cb_recommender.get_skill_gap_summary(user_id)
        total_missing = sum(len(gaps) for gaps in skill_gaps.values())
        
        # Format skill gap summary
        gap_summary = {}
        for category, gaps in skill_gaps.items():
            for gap in gaps[:3]:  # Top 3 per category
                level_desc = f"Level {gap.current_level}/5 → {gap.required_level}/5 needed"
                gap_summary[gap.skill_name] = level_desc
        
        return RecommendationResult(
            user_id=user_id,
            target_role=user.target_role,
            recommendations=recommendations,
            skill_gap_summary=gap_summary,
            total_missing_skills=total_missing,
            estimated_learning_hours=total_hours
        )
    
    def get_learning_path(
        self, 
        user_id: str, 
        max_courses: int = 10
    ) -> List[Course]:
        """
        Get ordered learning path using knowledge graph recommender.
        """
        return self.kg_recommender.get_learning_path(user_id, max_courses)
    
    def get_skill_gaps(self, user_id: str) -> Dict[str, List[SkillGap]]:
        """
        Get categorized skill gaps for a user.
        """
        return self.cb_recommender.get_skill_gap_summary(user_id)
    
    def get_next_skills(self, user_id: str, count: int = 5) -> List[Dict]:
        """
        Get next skills to learn in order.
        """
        return self.kg_recommender.get_next_skills(user_id, count)
    
    def explain_recommendation(
        self, 
        user_id: str, 
        course_id: str
    ) -> Dict:
        """
        Get detailed explanation for why a course was recommended.
        """
        if course_id not in self.courses:
            return {"error": "Course not found"}
        
        course = self.courses[course_id]
        user = self.users.get(user_id)
        
        if not user:
            return {"error": "User not found"}
        
        # Get individual scores
        cf_recs = self.cf_recommender.recommend(user_id, 50)
        cb_recs = self.cb_recommender.recommend(user_id, 50)
        kg_recs = self.kg_recommender.recommend(user_id, 50)
        
        cf_score = next((s for c, s, _ in cf_recs if c == course_id), None)
        cb_data = next(((s, r, sk) for c, s, r, sk in cb_recs if c == course_id), None)
        kg_data = next(((s, r, seq) for c, s, r, seq in kg_recs if c == course_id), None)
        
        weights = self._get_weights(user_id)
        
        explanation = {
            "course": {
                "id": course.id,
                "title": course.title,
                "skills_taught": course.skills_taught,
                "difficulty": course.difficulty.value,
                "duration_hours": course.duration_hours
            },
            "user_context": {
                "current_skills": list(user.current_skills.keys()),
                "target_role": user.target_role,
                "years_experience": user.years_experience
            },
            "scores": {
                "collaborative_filtering": {
                    "score": cf_score,
                    "weight": weights['collaborative'],
                    "reason": "Based on similar users' learning patterns"
                },
                "content_based": {
                    "score": cb_data[0] if cb_data else None,
                    "weight": weights['content_based'],
                    "skills_addressed": cb_data[2] if cb_data else [],
                    "reason": cb_data[1] if cb_data else "Not recommended by this method"
                },
                "knowledge_graph": {
                    "score": kg_data[0] if kg_data else None,
                    "weight": weights['knowledge_graph'],
                    "sequence_position": kg_data[2] if kg_data else None,
                    "reason": kg_data[1] if kg_data else "Not recommended by this method"
                }
            }
        }
        
        return explanation
