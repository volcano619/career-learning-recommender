"""
Data Loader for Career Learning Path Recommender System

This module handles loading and parsing all data files into structured models.
It provides a central interface for accessing:
- Course catalog
- User profiles
- User-course interactions
- Skills taxonomy
- Career paths

Author: RecommenderSystem
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from config import (
    COURSES_FILE, USERS_FILE, INTERACTIONS_FILE, 
    SKILLS_TAXONOMY_FILE, CAREER_PATHS_FILE
)
from models.data_models import Course, User, Interaction, CareerPath, Difficulty

logger = logging.getLogger(__name__)


def load_courses() -> Dict[str, Course]:
    """
    Load course catalog from JSON file.
    
    Returns:
        Dictionary mapping course_id to Course object
    """
    with open(COURSES_FILE, 'r') as f:
        data = json.load(f)
    
    courses = {}
    for course_data in data['courses']:
        # Convert difficulty string to enum
        course_data['difficulty'] = Difficulty(course_data['difficulty'])
        course = Course(**course_data)
        courses[course.id] = course
    
    logger.info(f"Loaded {len(courses)} courses")
    return courses


def load_users() -> Dict[str, User]:
    """
    Load user profiles from JSON file.
    
    Returns:
        Dictionary mapping user_id to User object
    """
    with open(USERS_FILE, 'r') as f:
        data = json.load(f)
    
    users = {}
    for user_data in data['users']:
        user = User(**user_data)
        users[user.id] = user
    
    logger.info(f"Loaded {len(users)} users")
    return users


def load_interactions() -> List[Interaction]:
    """
    Load user-course interactions from JSON file.
    
    Returns:
        List of Interaction objects
    """
    with open(INTERACTIONS_FILE, 'r') as f:
        data = json.load(f)
    
    interactions = [Interaction(**item) for item in data['interactions']]
    logger.info(f"Loaded {len(interactions)} interactions")
    return interactions


def load_skills_taxonomy() -> Dict:
    """
    Load skills taxonomy from JSON file.
    
    Returns:
        Dictionary with skill categories and skill definitions
    """
    with open(SKILLS_TAXONOMY_FILE, 'r') as f:
        data = json.load(f)
    
    # Flatten skills for easy lookup
    all_skills = {}
    for category_id, category_data in data['categories'].items():
        for skill_id, skill_data in category_data['skills'].items():
            all_skills[skill_id] = {
                'category': category_id,
                'category_name': category_data['name'],
                **skill_data
            }
    
    logger.info(f"Loaded {len(all_skills)} skills from taxonomy")
    return {
        'categories': data['categories'],
        'skills': all_skills
    }


def load_career_paths() -> Dict[str, CareerPath]:
    """
    Load career path definitions from JSON file.
    
    Returns:
        Dictionary mapping role_id to CareerPath object
    """
    with open(CAREER_PATHS_FILE, 'r') as f:
        data = json.load(f)
    
    career_paths = {}
    for role_id, role_data in data['roles'].items():
        career_path = CareerPath(**role_data)
        career_paths[role_id] = career_path
    
    logger.info(f"Loaded {len(career_paths)} career paths")
    return career_paths


def get_user_interaction_matrix(
    users: Dict[str, User],
    courses: Dict[str, Course],
    interactions: List[Interaction]
) -> Dict[str, Dict[str, float]]:
    """
    Build user-course interaction matrix for collaborative filtering.
    
    Values are based on:
    - Enrolled but not completed: 0.5
    - Completed: 0.8
    - Completed with rating: rating / 5
    
    Returns:
        Dictionary of {user_id: {course_id: score}}
    """
    matrix = {user_id: {} for user_id in users}
    
    for interaction in interactions:
        if interaction.user_id not in matrix:
            continue
            
        if interaction.rating is not None:
            score = interaction.rating / 5.0
        elif interaction.completed:
            score = 0.8
        else:
            score = 0.5
            
        matrix[interaction.user_id][interaction.course_id] = score
    
    return matrix


def get_course_skill_matrix(
    courses: Dict[str, Course],
    skills: Dict
) -> Dict[str, List[str]]:
    """
    Build course-to-skills mapping.
    
    Returns:
        Dictionary of {course_id: [skill_ids]}
    """
    return {course_id: course.skills_taught for course_id, course in courses.items()}


def load_all_data() -> Tuple[Dict[str, Course], Dict[str, User], List[Interaction], Dict, Dict[str, CareerPath]]:
    """
    Load all data files at once.
    
    Returns:
        Tuple of (courses, users, interactions, skills_taxonomy, career_paths)
    """
    courses = load_courses()
    users = load_users()
    interactions = load_interactions()
    skills_taxonomy = load_skills_taxonomy()
    career_paths = load_career_paths()
    
    return courses, users, interactions, skills_taxonomy, career_paths


if __name__ == "__main__":
    # Test data loading
    logging.basicConfig(level=logging.INFO)
    
    courses, users, interactions, skills, career_paths = load_all_data()
    
    print(f"\n📚 Courses: {len(courses)}")
    print(f"👤 Users: {len(users)}")
    print(f"🔗 Interactions: {len(interactions)}")
    print(f"🎯 Skills: {len(skills['skills'])}")
    print(f"💼 Career Paths: {len(career_paths)}")
    
    # Sample data
    sample_course = list(courses.values())[0]
    print(f"\nSample Course: {sample_course.title}")
    print(f"  Skills: {sample_course.skills_taught}")
    
    sample_user = list(users.values())[0]
    print(f"\nSample User: {sample_user.name}")
    print(f"  Target Role: {sample_user.target_role}")
    print(f"  Current Skills: {list(sample_user.current_skills.keys())}")
