"""
Configuration for Career Learning Path Recommender System

This module contains all configuration constants and settings used across
the recommendation system.

Author: RecommenderSystem
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Data files
COURSES_FILE = DATA_DIR / "courses.json"
USERS_FILE = DATA_DIR / "users.json"
INTERACTIONS_FILE = DATA_DIR / "interactions.json"
SKILLS_TAXONOMY_FILE = DATA_DIR / "skills_taxonomy.json"
CAREER_PATHS_FILE = DATA_DIR / "career_paths.json"

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
EMBEDDING_DIM = 384

# Collaborative filtering
CF_MIN_INTERACTIONS = 3  # Minimum interactions to use CF
CF_NUM_NEIGHBORS = 10    # Number of similar users/items to consider
CF_SIMILARITY_THRESHOLD = 0.1  # Minimum similarity to consider

# Content-based filtering
CB_SKILL_WEIGHT = 0.7    # Weight for skill match
CB_DIFFICULTY_WEIGHT = 0.2  # Weight for difficulty match
CB_POPULARITY_WEIGHT = 0.1  # Weight for course popularity

# Knowledge graph
KG_PREREQUISITE_PENALTY = 0.5  # Penalty for missing prerequisites
KG_PATH_BONUS = 0.3  # Bonus for courses on career path

# ============================================================================
# HYBRID WEIGHTS
# ============================================================================
# These weights determine how much each recommender contributes
# They are dynamically adjusted based on data availability

HYBRID_WEIGHTS = {
    "collaborative": 0.3,
    "content_based": 0.4,
    "knowledge_graph": 0.3
}

# Cold-start weights (when user has few interactions)
COLD_START_WEIGHTS = {
    "collaborative": 0.1,
    "content_based": 0.5,
    "knowledge_graph": 0.4
}

# ============================================================================
# RECOMMENDATION SETTINGS
# ============================================================================
DEFAULT_TOP_K = 10  # Default number of recommendations
MAX_TOP_K = 50      # Maximum recommendations to return
MIN_SCORE_THRESHOLD = 0.1  # Minimum score to include in recommendations

# Diversity settings
DIVERSITY_WEIGHT = 0.2  # How much to prioritize diversity
MAX_SAME_CATEGORY = 3   # Max courses from same category in top-K

# ============================================================================
# APP SETTINGS
# ============================================================================
APP_TITLE = "🎓 Career Learning Path Recommender"
APP_LAYOUT = "wide"
DEBUG_MODE = False

# Skill proficiency levels
PROFICIENCY_LEVELS = {
    1: "Beginner",
    2: "Elementary", 
    3: "Intermediate",
    4: "Advanced",
    5: "Expert"
}

# Course difficulty levels
DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced", "Expert"]

# ============================================================================
# DOMAINS
# ============================================================================
SKILL_DOMAINS = [
    "Data Science",
    "Machine Learning",
    "Cloud Computing",
    "DevOps",
    "Web Development",
    "Mobile Development",
    "Cybersecurity",
    "Database",
    "Programming Languages",
    "Soft Skills"
]
