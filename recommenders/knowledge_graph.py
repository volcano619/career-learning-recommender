"""
Knowledge Graph Recommender

This module implements career-path-aware recommendations using a knowledge graph approach.
It considers:
1. Skill prerequisites (learn X before Y)
2. Career path progression (skills needed for target role)
3. Learning path sequencing (ordered course recommendations)

WHY KNOWLEDGE GRAPH:
- Ensures logical learning progression
- Respects skill dependencies
- Creates coherent learning paths, not just isolated courses

Author: RecommenderSystem
"""

from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging

from models.data_models import Course, User, CareerPath, Difficulty
from config import KG_PREREQUISITE_PENALTY, KG_PATH_BONUS

logger = logging.getLogger(__name__)


class KnowledgeGraphRecommender:
    """
    Knowledge Graph Recommender using skill relationships and career paths.
    
    This recommender builds a graph of skill relationships and uses it to:
    1. Identify optimal learning sequences
    2. Penalize courses with unmet prerequisites
    3. Prioritize courses on the career path
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
        
        # Build skill graph
        self._build_skill_graph()
        
    def _build_skill_graph(self):
        """Build directed graph of skill prerequisites and relationships."""
        # Prerequisites: skill -> list of prerequisite skills
        self.prerequisites: Dict[str, List[str]] = {}
        # Dependents: skill -> list of skills that depend on it
        self.dependents: Dict[str, List[str]] = defaultdict(list)
        # Skill levels
        self.skill_levels: Dict[str, str] = {}
        
        for skill_id, skill_data in self.skills.items():
            prereqs = skill_data.get('prerequisites', [])
            self.prerequisites[skill_id] = prereqs
            self.skill_levels[skill_id] = skill_data.get('level', 'intermediate')
            
            for prereq in prereqs:
                self.dependents[prereq].append(skill_id)
        
        logger.info(f"Built skill graph with {len(self.prerequisites)} skills")
    
    def _get_missing_prerequisites(
        self, 
        skill_id: str, 
        user_skills: Set[str]
    ) -> List[str]:
        """
        Find prerequisites for a skill that the user doesn't have.
        
        Args:
            skill_id: Target skill
            user_skills: Set of skills user already has
            
        Returns:
            List of missing prerequisite skill IDs
        """
        prereqs = self.prerequisites.get(skill_id, [])
        return [p for p in prereqs if p not in user_skills]
    
    def _get_skill_depth(self, skill_id: str, memo: Dict[str, int] = None) -> int:
        """
        Calculate the depth of a skill in the prerequisite graph.
        Depth 0 = no prerequisites, higher = more advanced.
        """
        if memo is None:
            memo = {}
        
        if skill_id in memo:
            return memo[skill_id]
        
        prereqs = self.prerequisites.get(skill_id, [])
        if not prereqs:
            memo[skill_id] = 0
            return 0
        
        max_prereq_depth = max(self._get_skill_depth(p, memo) for p in prereqs)
        memo[skill_id] = max_prereq_depth + 1
        return memo[skill_id]
    
    def _get_learning_path_order(
        self, 
        target_skills: List[str], 
        user_skills: Set[str]
    ) -> List[str]:
        """
        Order skills by their logical learning sequence.
        
        Uses topological sort considering prerequisites.
        
        Args:
            target_skills: Skills to learn
            user_skills: Skills user already has
            
        Returns:
            Ordered list of skills to learn
        """
        # Expand to include prerequisites
        all_skills_needed = set()
        stack = list(target_skills)
        
        while stack:
            skill = stack.pop()
            if skill in user_skills or skill in all_skills_needed:
                continue
            
            all_skills_needed.add(skill)
            prereqs = self.prerequisites.get(skill, [])
            stack.extend(prereqs)
        
        # Sort by depth (learn fundamental skills first)
        depth_memo = {}
        skills_with_depth = [
            (skill, self._get_skill_depth(skill, depth_memo))
            for skill in all_skills_needed
        ]
        skills_with_depth.sort(key=lambda x: x[1])
        
        return [skill for skill, _ in skills_with_depth]
    
    def _calculate_prerequisite_score(
        self, 
        course: Course, 
        user_skills: Set[str]
    ) -> Tuple[float, List[str]]:
        """
        Calculate how ready the user is to take this course.
        
        Returns:
            Tuple of (readiness_score, missing_prerequisites)
        """
        missing_prereqs = set()
        
        for skill in course.skills_taught:
            missing = self._get_missing_prerequisites(skill, user_skills)
            missing_prereqs.update(missing)
        
        if not missing_prereqs:
            return 1.0, []
        
        # Penalize based on number of missing prerequisites
        penalty = KG_PREREQUISITE_PENALTY * len(missing_prereqs)
        score = max(0.1, 1.0 - penalty)
        
        return score, list(missing_prereqs)
    
    def _calculate_path_alignment_score(
        self, 
        course: Course, 
        career_path: CareerPath
    ) -> float:
        """
        Calculate how well course aligns with career path.
        
        Returns:
            Score between 0 and 1
        """
        required_skills = set(career_path.required_skills)
        nice_to_have = set(career_path.nice_to_have)
        course_skills = set(course.skills_taught)
        
        required_overlap = len(course_skills & required_skills)
        nice_overlap = len(course_skills & nice_to_have)
        
        if not course_skills:
            return 0.0
        
        # Weight required skills higher
        score = (required_overlap * 1.0 + nice_overlap * 0.5) / len(course_skills)
        return min(1.0, score + KG_PATH_BONUS if required_overlap > 0 else score)
    
    def recommend(
        self, 
        user_id: str, 
        top_k: int = 10
    ) -> List[Tuple[str, float, str, int]]:
        """
        Generate knowledge-graph-aware recommendations.
        
        Args:
            user_id: Target user ID
            top_k: Number of recommendations to return
            
        Returns:
            List of (course_id, score, reason, sequence_order) tuples
        """
        if user_id not in self.users:
            logger.warning(f"User {user_id} not found")
            return []
        
        user = self.users[user_id]
        user_skills = set(user.current_skills.keys())
        
        if user.target_role not in self.career_paths:
            logger.warning(f"Career path {user.target_role} not found")
            return []
        
        career_path = self.career_paths[user.target_role]
        
        # Get ordered learning path
        target_skills = career_path.required_skills + career_path.nice_to_have
        ordered_skills = self._get_learning_path_order(target_skills, user_skills)
        
        # Create skill priority map (earlier in path = higher priority)
        skill_priority = {skill: i for i, skill in enumerate(ordered_skills)}
        
        # Score each course
        course_scores: Dict[str, Tuple[float, str, int]] = {}
        
        for course_id, course in self.courses.items():
            # Check if course teaches any needed skills
            teaches_needed = [s for s in course.skills_taught if s in skill_priority]
            if not teaches_needed:
                continue
            
            # Calculate scores
            prereq_score, missing = self._calculate_prerequisite_score(course, user_skills)
            path_score = self._calculate_path_alignment_score(course, career_path)
            
            # Prioritize courses for earlier skills in the learning path
            earliest_skill = min(skill_priority.get(s, 999) for s in teaches_needed)
            sequence_score = 1.0 / (1 + earliest_skill * 0.1)  # Higher for earlier skills
            
            # Combined score
            total_score = 0.4 * prereq_score + 0.3 * path_score + 0.3 * sequence_score
            
            # Generate reason
            if missing:
                missing_names = [self.skills.get(m, {}).get('name', m) for m in missing[:2]]
                reason = f"Learn after: {', '.join(missing_names)}"
            else:
                taught_names = [self.skills.get(s, {}).get('name', s) for s in teaches_needed[:2]]
                reason = f"Next step: teaches {', '.join(taught_names)}"
            
            course_scores[course_id] = (total_score, reason, earliest_skill)
        
        # Sort by score, then by sequence order
        sorted_courses = sorted(
            course_scores.items(), 
            key=lambda x: (-x[1][0], x[1][2])  # Score desc, then sequence asc
        )
        
        return [(cid, score, reason, seq) for cid, (score, reason, seq) in sorted_courses[:top_k]]
    
    def get_learning_path(
        self, 
        user_id: str, 
        max_courses: int = 10
    ) -> List[Course]:
        """
        Generate an ordered learning path for the user.
        
        Returns courses in the recommended order to take them.
        """
        recommendations = self.recommend(user_id, max_courses)
        
        # Sort by sequence order
        sorted_recs = sorted(recommendations, key=lambda x: x[3])
        
        return [self.courses[course_id] for course_id, _, _, _ in sorted_recs]
    
    def get_next_skills(self, user_id: str, count: int = 5) -> List[Dict]:
        """
        Get the next skills the user should learn in order.
        
        Returns:
            List of skill dictionaries with info
        """
        if user_id not in self.users:
            return []
        
        user = self.users[user_id]
        user_skills = set(user.current_skills.keys())
        
        if user.target_role not in self.career_paths:
            return []
        
        career_path = self.career_paths[user.target_role]
        target_skills = career_path.required_skills + career_path.nice_to_have
        ordered_skills = self._get_learning_path_order(target_skills, user_skills)
        
        result = []
        for skill_id in ordered_skills[:count]:
            skill_data = self.skills.get(skill_id, {})
            missing_prereqs = self._get_missing_prerequisites(skill_id, user_skills)
            
            result.append({
                'skill_id': skill_id,
                'name': skill_data.get('name', skill_id),
                'level': skill_data.get('level', 'intermediate'),
                'missing_prerequisites': missing_prereqs,
                'ready_to_learn': len(missing_prereqs) == 0
            })
        
        return result
