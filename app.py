"""
Streamlit App for Career Learning Path Recommender System

This module provides an interactive web interface for the recommender system.
Features:
1. User profile selection and creation
2. Skill gap visualization
3. Personalized course recommendations with explanations
4. Learning path view
5. Recommendation explainability

Author: RecommenderSystem
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import logging

from config import APP_TITLE, APP_LAYOUT, PROFICIENCY_LEVELS, DIFFICULTY_LEVELS
from data_loader import load_all_data
from recommenders.hybrid import HybridRecommender
from models.data_models import Recommendation
import shared_ui

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title=APP_TITLE, 
    layout=APP_LAYOUT,
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# Apply global theme
shared_ui.apply_global_theme()

# Custom project-specific CSS extensions
st.markdown("""
<style>
    .skill-gap-critical {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 12px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .skill-gap-important {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .course-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .course-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)




# ============================================================================
# INITIALIZATION (Cached)
# ============================================================================

@st.cache_resource
def initialize_system():
    """Initialize recommender system (cached to avoid reloading)."""
    with st.spinner("🚀 Initializing Recommender System..."):
        logger.info("Loading data and initializing recommender...")
        
        # Load all data
        courses, users, interactions, skills_taxonomy, career_paths = load_all_data()
        
        # Initialize hybrid recommender
        recommender = HybridRecommender(
            courses=courses,
            users=users,
            interactions=interactions,
            career_paths=career_paths,
            skills_taxonomy=skills_taxonomy
        )
        
        return recommender, courses, users, career_paths, skills_taxonomy


# Initialize system
try:
    recommender, courses, users, career_paths, skills_taxonomy = initialize_system()
    system_ready = True
except Exception as e:
    st.error(f"❌ System Initialization Failed: {e}")
    system_ready = False
    st.stop()


# ============================================================================
# SIDEBAR - User Selection
# ============================================================================

with st.sidebar:
    st.markdown("## 👤 User Profile")
    
    # Help Section
    shared_ui.add_help_section(
        "Career Recommender",
        "Personalized career path generator and skill gap analyzer.",
        "Select your target role and current skills. The AI will map your journey and recommend resources.",
        "Traditional job boards show roles; this system identifies the EXACT skill gaps and provides a learning roadmap.",
        "LinkedIn says a job needs 'Python'; this app tells you 'You need Pandas/NumPy' and gives you a 1-month plan."
    )
    
    # User selection
    user_options = {user.id: f"{user.name} ({user.current_role})" for user in users.values()}
    if "selected_user_id" not in st.session_state:
        st.session_state.selected_user_id = list(user_options.keys())[0]
    selected_user_id = st.selectbox(
        "Select User",
        options=list(user_options.keys()),
        format_func=lambda x: user_options[x],
        key="selected_user_id",
        help="Select a user profile to see personalized recommendations"
    )
    
    selected_user = users[selected_user_id]
    
    # User info display
    st.markdown("---")
    st.markdown("### 📋 Profile Details")
    
    col1, col2 = st.columns(2)
    with col1:
        shared_ui.create_metric_card("Experience", f"{selected_user.years_experience} yrs")
    with col2:
        shared_ui.create_metric_card("Weekly Hours", f"{selected_user.weekly_hours}h")
    
    st.markdown(f"**Current Role:** {selected_user.current_role}")
    
    target_role_name = career_paths.get(selected_user.target_role, None)
    if target_role_name:
        st.markdown(f"**Target Role:** {target_role_name.name}")
    
    st.markdown(f"**Learning Style:** {selected_user.learning_style.value.title()}")
    
    # Current skills
    st.markdown("---")
    st.markdown("### 🎯 Current Skills")
    
    if selected_user.current_skills:
        for skill_id, level in list(selected_user.current_skills.items())[:8]:
            skill_name = skills_taxonomy['skills'].get(skill_id, {}).get('name', skill_id)
            st.progress(level / 5, text=f"{skill_name}: {PROFICIENCY_LEVELS.get(level, 'Unknown')}")
    else:
        st.info("No skills recorded yet")
    
    # Settings
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    if "rec_top_k" not in st.session_state:
        from config import DEFAULT_TOP_K
        st.session_state.rec_top_k = DEFAULT_TOP_K
    top_k = st.slider("Number of Recommendations", 3, 15, key="rec_top_k")
    
    if "show_explanations" not in st.session_state:
        st.session_state.show_explanations = True
    show_explanations = st.checkbox("Show Detailed Explanations", key="show_explanations")


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
shared_ui.add_header(
    "🎓 Career Recommender System",
    "Personalized learning paths to accelerate your career growth | *Driving 5-15% increase in course completions*"
)

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Skill Gap Analysis", 
    "📚 Recommendations", 
    "🛤️ Learning Path",
    "🔍 Explore Courses"
])


# ============================================================================
# TAB 1: Skill Gap Analysis
# ============================================================================

with tab1:
    st.markdown("## Skill Gap Analysis")
    st.markdown(f"Comparing your current skills with requirements for **{career_paths.get(selected_user.target_role, {}).name if selected_user.target_role in career_paths else 'Unknown Role'}**")
    
    # Get skill gaps
    skill_gaps = recommender.get_skill_gaps(selected_user_id)
    
    if skill_gaps:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        critical_count = len(skill_gaps.get('critical', []))
        important_count = len(skill_gaps.get('important', []))
        nice_count = len(skill_gaps.get('nice_to_have', []))
        total_gaps = critical_count + important_count + nice_count
        
        with col1:
            shared_ui.create_metric_card("Total Skill Gaps", str(total_gaps))
        with col2:
            shared_ui.create_metric_card("Critical", str(critical_count), delta_pos=False if critical_count > 0 else True)
        with col3:
            shared_ui.create_metric_card("Important", str(important_count))
        with col4:
            shared_ui.create_metric_card("Nice to Have", str(nice_count))
        
        st.markdown("---")
        
        # Skill gap visualization
        col_chart, col_details = st.columns([2, 1])
        
        with col_chart:
            # Create radar chart for skill comparison
            all_gaps = (
                skill_gaps.get('critical', []) + 
                skill_gaps.get('important', []) + 
                skill_gaps.get('nice_to_have', [])
            )
            
            if all_gaps:
                # Prepare data for radar chart
                skill_names = [gap.skill_name for gap in all_gaps[:10]]
                current_levels = [gap.current_level for gap in all_gaps[:10]]
                required_levels = [gap.required_level for gap in all_gaps[:10]]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=current_levels + [current_levels[0]],
                    theta=skill_names + [skill_names[0]],
                    fill='toself',
                    name='Current Level',
                    line_color='#667eea'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=required_levels + [required_levels[0]],
                    theta=skill_names + [skill_names[0]],
                    fill='toself',
                    name='Required Level',
                    line_color='#764ba2',
                    opacity=0.5
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 5])
                    ),
                    showlegend=True,
                    title="Skills Comparison: Current vs Required",
                    height=450
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col_details:
            st.markdown("### 🚨 Critical Skills")
            for gap in skill_gaps.get('critical', [])[:5]:
                st.markdown(f"""
                <div class="skill-gap-critical">
                    <strong>{gap.skill_name}</strong><br>
                    Level {gap.current_level} → {gap.required_level} needed
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### ⚠️ Important Skills")
            for gap in skill_gaps.get('important', [])[:3]:
                st.markdown(f"""
                <div class="skill-gap-important">
                    <strong>{gap.skill_name}</strong><br>
                    Level {gap.current_level} → {gap.required_level} needed
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("🎉 Congratulations! You have all the skills needed for your target role!")


# ============================================================================
# TAB 2: Recommendations
# ============================================================================

with tab2:
    st.markdown("## 📚 Personalized Course Recommendations")
    
    # Get recommendations
    with st.spinner("Generating personalized recommendations..."):
        result = recommender.recommend(selected_user_id, top_k)
    
    if result.recommendations:
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            shared_ui.create_metric_card("Recommendations", str(len(result.recommendations)), delta="Personalized", delta_pos=True)
        with col2:
            shared_ui.create_metric_card("Learning Effort", f"{result.estimated_learning_hours}h", delta="Total time")
        with col3:
            shared_ui.create_metric_card("Skills Covered", str(result.total_missing_skills), delta="Target Role Fit")
        
        st.markdown("---")
        
        # Display recommendations
        for rec in result.recommendations:
            with st.container():
                col_main, col_score = st.columns([4, 1])
                
                with col_main:
                    st.markdown(f"### {rec.rank}. {rec.course.title}")
                    
                    # Course metadata
                    meta_cols = st.columns(4)
                    with meta_cols[0]:
                        st.caption(f"📚 {rec.course.provider}")
                    with meta_cols[1]:
                        st.caption(f"⏱️ {rec.course.duration_hours} hours")
                    with meta_cols[2]:
                        st.caption(f"📊 {rec.course.difficulty.value}")
                    with meta_cols[3]:
                        st.caption(f"⭐ {rec.course.rating}/5")
                    
                    # Skills taught
                    skills_display = ", ".join(
                        skills_taxonomy['skills'].get(s, {}).get('name', s) 
                        for s in rec.course.skills_taught[:5]
                    )
                    st.markdown(f"**Skills:** {skills_display}")
                    
                    # Reasons (if enabled)
                    if show_explanations and rec.reasons:
                        with st.expander("💡 Why this recommendation?"):
                            for reason in rec.reasons:
                                st.markdown(f"• {reason}")
                            
                            if rec.skill_gaps_addressed:
                                addressed = ", ".join(
                                    skills_taxonomy['skills'].get(s, {}).get('name', s)
                                    for s in rec.skill_gaps_addressed[:5]
                                )
                                st.markdown(f"**Skills addressed:** {addressed}")
                
                with col_score:
                    # Score visualization
                    score_pct = int(rec.score * 100)
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px;">
                        <div style="font-size: 2rem; font-weight: bold; color: #667eea;">
                            {score_pct}%
                        </div>
                        <div style="font-size: 0.8rem; color: #666;">
                            Match Score
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
    else:
        st.info("No recommendations available. This could be because you already have all necessary skills!")


# ============================================================================
# TAB 3: Learning Path
# ============================================================================

with tab3:
    st.markdown("## 🛤️ Personalized Learning Path")
    st.markdown("Courses ordered by the optimal learning sequence")
    
    # Get learning path
    learning_path = recommender.get_learning_path(selected_user_id, max_courses=10)
    next_skills = recommender.get_next_skills(selected_user_id, count=8)
    
    if next_skills:
        st.markdown("### 📈 Skills Progression")
        
        # Create timeline visualization
        fig = go.Figure()
        
        for i, skill in enumerate(next_skills):
            color = '#4caf50' if skill['ready_to_learn'] else '#ff9800'
            symbol = '✓ Ready' if skill['ready_to_learn'] else f"⏳ After: {', '.join(skill['missing_prerequisites'][:2])}"
            
            fig.add_trace(go.Scatter(
                x=[i],
                y=[1],
                mode='markers+text',
                marker=dict(size=40, color=color),
                text=[skill['name']],
                textposition='top center',
                hovertemplate=f"{skill['name']}<br>{symbol}<extra></extra>"
            ))
        
        fig.update_layout(
            showlegend=False,
            height=200,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 2]),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        # Add connecting line
        fig.add_shape(
            type="line",
            x0=0, y0=1, x1=len(next_skills)-1, y1=1,
            line=dict(color="#e0e0e0", width=3)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Skills table
        skills_df = pd.DataFrame([
            {
                'Order': i + 1,
                'Skill': s['name'],
                'Level': s['level'].title(),
                'Status': '✅ Ready' if s['ready_to_learn'] else '⏳ Pending',
                'Prerequisites': ', '.join(s['missing_prerequisites']) if s['missing_prerequisites'] else '-'
            }
            for i, s in enumerate(next_skills)
        ])
        st.dataframe(skills_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 📚 Recommended Course Sequence")
    
    if learning_path:
        for i, course in enumerate(learning_path, 1):
            col1, col2 = st.columns([1, 15])
            
            with col1:
                st.markdown(f"""
                <div style="
                    width: 40px; height: 40px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    color: white; font-weight: bold;
                ">
                    {i}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                with st.expander(f"**{course.title}** - {course.duration_hours}h"):
                    st.markdown(f"**Provider:** {course.provider}")
                    st.markdown(f"**Difficulty:** {course.difficulty.value}")
                    st.markdown(f"**Rating:** {'⭐' * int(course.rating)} ({course.rating})")
                    
                    skills_list = [
                        skills_taxonomy['skills'].get(s, {}).get('name', s) 
                        for s in course.skills_taught
                    ]
                    st.markdown(f"**Skills:** {', '.join(skills_list)}")
    else:
        st.info("No learning path available.")


# ============================================================================
# TAB 4: Explore Courses
# ============================================================================

with tab4:
    st.markdown("## 🔍 Explore All Courses")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categories = list(set(c.category for c in courses.values()))
        selected_category = st.selectbox("Category", ["All"] + sorted(categories))
    
    with col2:
        selected_difficulty = st.selectbox("Difficulty", ["All"] + DIFFICULTY_LEVELS)
    
    with col3:
        sort_by = st.selectbox("Sort By", ["Rating", "Duration", "Reviews"])
    
    # Filter courses
    filtered_courses = list(courses.values())
    
    if selected_category != "All":
        filtered_courses = [c for c in filtered_courses if c.category == selected_category]
    
    if selected_difficulty != "All":
        filtered_courses = [c for c in filtered_courses if c.difficulty.value == selected_difficulty]
    
    # Sort
    if sort_by == "Rating":
        filtered_courses.sort(key=lambda x: x.rating, reverse=True)
    elif sort_by == "Duration":
        filtered_courses.sort(key=lambda x: x.duration_hours)
    elif sort_by == "Reviews":
        filtered_courses.sort(key=lambda x: x.num_reviews, reverse=True)
    
    # Display
    st.markdown(f"**Showing {len(filtered_courses)} courses**")
    
    # Create grid
    cols_per_row = 3
    for i in range(0, len(filtered_courses), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(filtered_courses):
                course = filtered_courses[i + j]
                
                with col:
                    with st.container():
                        st.markdown(f"**{course.title}**")
                        st.caption(f"{course.provider} | {course.difficulty.value}")
                        st.caption(f"⏱️ {course.duration_hours}h | ⭐ {course.rating}")
                        
                        skills = course.skills_taught[:3]
                        skill_names = [skills_taxonomy['skills'].get(s, {}).get('name', s) for s in skills]
                        st.caption(f"Skills: {', '.join(skill_names)}")
                        
                        # Check if this course is recommended
                        is_recommended = course.id in [r.course.id for r in result.recommendations]
                        if is_recommended:
                            st.success("✨ Recommended for you!")
                        
                        st.markdown("---")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    🎓 Career Learning Path Recommender | Powered by Hybrid ML Recommendations<br>
    Combining Collaborative Filtering + Content-Based + Knowledge Graph approaches
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.875rem; padding: 2rem 0;">
    🎓 Career Recommender System | Built with Hybrid Filtering + Knowledge Graphs<br>
    <span style="font-family: 'Roboto Mono', monospace;">Version 1.2.0-Premium</span>
</div>
""", unsafe_allow_html=True)
