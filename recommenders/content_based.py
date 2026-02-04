"""
Content-Based Filtering Recommender

This module implements content-based filtering for course recommendations.
It recommends courses based on:
1. Skill gap analysis (missing skills for target role)
2. Skill matching (courses that teach needed skills)
3. User preferences (difficulty level, interests)

WHY CONTENT-BASED FILTERING:
- Works with new users (no cold-start problem)
- Directly addresses skill gaps
- Explainable recommendations ("this course teaches X which you need")

Author: RecommenderSystem
"""

from typing import Dict, List, Tuple, Set
import logging

from models.data_models import Course, User, CareerPath, SkillGap, Difficulty
from config import CB_SKILL_WEIGHT, CB_DIFFICULTY_WEIGHT, CB_POPULARITY_WEIGHT

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Content-Based Recommender using skill matching and gap analysis.
    """
    
    def __init__(
        self,
        courses: Dict[str, Course],
        users: Dict[str, User],
        career_paths: Dict[str, CareerPath],
        skills_taxonomy: Dict
    ):
        self.courses = courses
        self.users = users
        self.career_paths = career_paths
        self.skills = skills_taxonomy['skills']
        
        # Build course skill index for fast lookup
        self._build_skill_index()
    
    def _build_skill_index(self):
        """Build inverted index: skill -> courses that teach it."""
        self.skill_to_courses: Dict[str, List[str]] = {}
        
        for course_id, course in self.courses.items():
            for skill in course.skills_taught:
                if skill not in self.skill_to_courses:
                    self.skill_to_courses[skill] = []
                self.skill_to_courses[skill].append(course_id)
        
        logger.info(f"Built skill index with {len(self.skill_to_courses)} skills")
    
    def analyze_skill_gap(
        self, 
        user: User, 
        career_path: CareerPath
    ) -> List[SkillGap]:
        """
        Analyze the skill gap between user's current skills and target role requirements.
        
        Args:
            user: User profile
            career_path: Target career path
            
        Returns:
            List of SkillGap objects, sorted by importance
        """
        skill_gaps = []
        
        # Get required skills with importance weights
        skill_importance = career_path.skill_importance
        all_required = set(career_path.required_skills + career_path.nice_to_have)
        
        for skill_id in all_required:
            current_level = user.current_skills.get(skill_id, 0)
            
            # Determine required level based on importance
            importance = skill_importance.get(skill_id, 0.5)
            if importance >= 0.9:
                required_level = 5
            elif importance >= 0.7:
                required_level = 4
            elif importance >= 0.5:
                required_level = 3
            else:
                required_level = 2
            
            # Only add as gap if there's actually a gap
            if current_level < required_level:
                skill_name = self.skills.get(skill_id, {}).get('name', skill_id)
                gap = SkillGap(
                    skill_id=skill_id,
                    skill_name=skill_name,
                    current_level=current_level,
                    required_level=required_level,
                    importance=importance,
                    gap_size=required_level - current_level
                )
                skill_gaps.append(gap)
        
        # Sort by importance (descending)
        skill_gaps.sort(key=lambda x: (x.importance, x.gap_size), reverse=True)
        
        logger.info(f"Found {len(skill_gaps)} skill gaps for user {user.id}")
        return skill_gaps
    
    def _get_difficulty_score(self, user: User, course: Course) -> float:
        """
        Calculate difficulty appropriateness score.
        
        Logic:
        - If user is beginner and course is advanced, lower score
        - If user is advanced and course is beginner, lower score
        - Sweet spot: course is one level above user's average
        """
        # Estimate user level from average skill proficiency
        if user.current_skills:
            avg_proficiency = sum(user.current_skills.values()) / len(user.current_skills)
        else:
            avg_proficiency = 1.0
        
        # Map user proficiency to difficulty
        user_level = min(3, int(avg_proficiency))  # 0, 1, 2, 3
        
        difficulty_order = {
            Difficulty.BEGINNER: 0,
            Difficulty.INTERMEDIATE: 1,
            Difficulty.ADVANCED: 2,
            Difficulty.EXPERT: 3
        }
        course_level = difficulty_order.get(course.difficulty, 1)
        
        # Ideal: course is same level or one level up
        level_diff = abs(course_level - user_level)
        
        if course_level == user_level + 1:
            return 1.0  # Ideal stretch
        elif course_level == user_level:
            return 0.9  # Good match
        elif level_diff == 1:
            return 0.7  # Acceptable
        elif level_diff == 2:
            return 0.4  # Less ideal
        else:
            return 0.2  # Too far
    
    def _get_popularity_score(self, course: Course) -> float:
        """Normalize popularity score based on reviews and rating."""
        # Normalize reviews (log scale)
        import math
        review_score = min(1.0, math.log10(course.num_reviews + 1) / 5)
        rating_score = course.rating / 5.0
        
        return 0.5 * review_score + 0.5 * rating_score
    
    def recommend(
        self, 
        user_id: str, 
        top_k: int = 10,
        exclude_user_skills: bool = True
    ) -> List[Tuple[str, float, str, List[str]]]:
        """
        Generate content-based recommendations for a user.
        
        Args:
            user_id: Target user ID
            top_k: Number of recommendations to return
            exclude_user_skills: Whether to exclude courses teaching only skills user has
            
        Returns:
            List of (course_id, score, reason, skills_addressed) tuples
        """
        if user_id not in self.users:
            logger.warning(f"User {user_id} not found")
            return []
        
        user = self.users[user_id]
        
        # Get career path
        if user.target_role not in self.career_paths:
            logger.warning(f"Career path {user.target_role} not found")
            return []
        
        career_path = self.career_paths[user.target_role]
        
        # Analyze skill gaps
        skill_gaps = self.analyze_skill_gap(user, career_path)
        missing_skills = {gap.skill_id: gap for gap in skill_gaps}
        
        if not missing_skills:
            logger.info(f"User {user_id} has no skill gaps for {user.target_role}")
            return []
        
        # Score each course
        course_scores: Dict[str, Tuple[float, str, List[str]]] = {}
        
        for course_id, course in self.courses.items():
            # Find skills this course teaches that user needs
            skills_addressed = [
                skill for skill in course.skills_taught 
                if skill in missing_skills
            ]
            
            if not skills_addressed:
                continue
            
            # Calculate skill match score
            skill_score = 0.0
            for skill_id in skills_addressed:
                gap = missing_skills[skill_id]
                # Weight by importance and gap size
                skill_score += gap.importance * (gap.gap_size / 5.0)
            
            # Normalize by number of skills
            skill_score = min(1.0, skill_score / len(skills_addressed))
            
            # Calculate other scores
            difficulty_score = self._get_difficulty_score(user, course)
            popularity_score = self._get_popularity_score(course)
            
            # Weighted combination
            total_score = (
                CB_SKILL_WEIGHT * skill_score +
                CB_DIFFICULTY_WEIGHT * difficulty_score +
                CB_POPULARITY_WEIGHT * popularity_score
            )
            
            # Generate reason
            skill_names = [missing_skills[s].skill_name for s in skills_addressed[:3]]
            if len(skills_addressed) > 3:
                reason = f"Teaches {', '.join(skill_names)} +{len(skills_addressed)-3} more needed skills"
            else:
                reason = f"Teaches {', '.join(skill_names)} which you need for {career_path.name}"
            
            course_scores[course_id] = (total_score, reason, skills_addressed)
        
        # Sort and return top K
        sorted_courses = sorted(course_scores.items(), key=lambda x: x[1][0], reverse=True)
        return [(cid, score, reason, skills) for cid, (score, reason, skills) in sorted_courses[:top_k]]
    
    def get_skill_gap_summary(self, user_id: str) -> Dict[str, List[SkillGap]]:
        """
        Get categorized skill gaps for a user.
        
        Returns:
            Dictionary with 'critical', 'important', 'nice_to_have' skill gap lists
        """
        if user_id not in self.users:
            return {}
        
        user = self.users[user_id]
        
        if user.target_role not in self.career_paths:
            return {}
        
        career_path = self.career_paths[user.target_role]
        skill_gaps = self.analyze_skill_gap(user, career_path)
        
        categorized = {
            'critical': [],
            'important': [],
            'nice_to_have': []
        }
        
        for gap in skill_gaps:
            if gap.importance >= 0.85:
                categorized['critical'].append(gap)
            elif gap.importance >= 0.6:
                categorized['important'].append(gap)
            else:
                categorized['nice_to_have'].append(gap)
        
        return categorized
