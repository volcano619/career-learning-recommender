"""
Data Models for Career Learning Path Recommender System

This module defines Pydantic models for all data entities used in the system.
Using Pydantic provides:
1. Data validation
2. Type hints
3. Easy serialization/deserialization
4. Self-documenting code

Author: RecommenderSystem
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum


class Difficulty(str, Enum):
    """Course difficulty levels"""
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"


class LearningStyle(str, Enum):
    """User learning style preferences"""
    VISUAL = "visual"
    HANDS_ON = "hands-on"
    READING = "reading"
    VIDEO = "video"
    PROJECT_BASED = "project-based"
    STRUCTURED = "structured"
    RESEARCH = "research"


class Skill(BaseModel):
    """Individual skill definition from taxonomy"""
    name: str
    level: str  # fundamental, intermediate, advanced, expert
    related: List[str] = []
    prerequisites: List[str] = []


class Course(BaseModel):
    """Course in the catalog"""
    id: str
    title: str
    provider: str
    skills_taught: List[str]
    difficulty: Difficulty
    duration_hours: int
    rating: float = Field(ge=0, le=5)
    num_reviews: int
    price: float = Field(ge=0)
    category: str
    description: str
    url: str
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Course):
            return self.id == other.id
        return False


class User(BaseModel):
    """User profile"""
    id: str
    name: str
    current_role: str
    target_role: str
    years_experience: int = Field(ge=0)
    current_skills: Dict[str, int]  # skill_id -> proficiency (1-5)
    learning_style: LearningStyle
    weekly_hours: int = Field(ge=0)
    interests: List[str] = []


class Interaction(BaseModel):
    """User-course interaction"""
    user_id: str
    course_id: str
    enrolled: bool = True
    completed: bool = False
    rating: Optional[float] = Field(default=None, ge=1, le=5)
    timestamp: str


class CareerPath(BaseModel):
    """Career role definition"""
    name: str
    description: str
    required_skills: List[str]
    nice_to_have: List[str] = []
    skill_importance: Dict[str, float] = {}  # skill -> importance weight
    salary_range: str = ""
    growth_roles: List[str] = []


class Recommendation(BaseModel):
    """A single course recommendation with explanation"""
    course: Course
    score: float = Field(ge=0, le=1)
    rank: int
    reasons: List[str] = []
    skill_gaps_addressed: List[str] = []
    source: str = "hybrid"  # collaborative, content_based, knowledge_graph, hybrid


class RecommendationResult(BaseModel):
    """Complete recommendation result for a user"""
    user_id: str
    target_role: str
    recommendations: List[Recommendation]
    skill_gap_summary: Dict[str, str] = {}  # skill -> current level description
    total_missing_skills: int = 0
    estimated_learning_hours: int = 0


class SkillGap(BaseModel):
    """Skill gap analysis for a user"""
    skill_id: str
    skill_name: str
    current_level: int  # 0 if user doesn't have it
    required_level: int  # Based on target role
    importance: float  # How important for target role
    gap_size: int  # required - current
