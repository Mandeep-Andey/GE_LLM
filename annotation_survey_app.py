"""
Annotation Survey App - Crowdsourced character interaction annotation.

A Streamlit-based survey app that presents dialogue snippets from Middlemarch
and asks annotators to identify character interactions.

Run with:
    streamlit run annotation_survey_app.py

Designed to be shareable with Digital Humanities communities for 
building gold-standard annotations.
"""

import streamlit as st
import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import random

# --- Configuration ---
DATA_DIR = Path("./data/Middlemarch")
ANNOTATIONS_DIR = Path("./crowd_annotations")
ANNOTATIONS_DIR.mkdir(exist_ok=True)

# Character list for validation
CHARACTER_FILE = Path("./char_alias.json")

# Interaction types
INTERACTION_TYPES = [
    "Direct Dialogue",
    "Physical Action", 
    "Observation",
    "Memory/Reference",
    "No Interaction"  # Important: allow annotators to say "no interaction here"
]


def load_characters() -> List[str]:
    """Load canonical character names."""
    if CHARACTER_FILE.exists():
        with open(CHARACTER_FILE, 'r') as f:
            data = json.load(f)
        return [char['canonical_name'] for char in data]
    return []


def load_snippets_for_annotation() -> List[Dict[str, Any]]:
    """
    Load text snippets that need annotation.
    
    These are pre-selected paragraphs containing potential interactions.
    In production, these would be generated from the extraction pipeline.
    """
    snippets_file = Path("./annotation_snippets.json")
    
    if snippets_file.exists():
        with open(snippets_file, 'r') as f:
            return json.load(f)
    
    # Generate sample snippets from book text
    snippets = []
    for book_file in sorted(DATA_DIR.glob("*.txt")):
        with open(book_file, 'r') as f:
            text = f.read()
        
        # Split into paragraphs and sample some
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        sampled = random.sample(paragraphs, min(5, len(paragraphs)))
        
        for i, para in enumerate(sampled):
            snippets.append({
                'id': f"{book_file.stem}_{i:03d}",
                'source': book_file.stem,
                'text': para[:1000],  # Limit to 1000 chars
                'context': f"From {book_file.stem}"
            })
    
    # Save for consistency
    with open(snippets_file, 'w') as f:
        json.dump(snippets, f, indent=2)
    
    return snippets


def get_annotator_id() -> str:
    """Get or create a unique annotator ID."""
    if 'annotator_id' not in st.session_state:
        # Generate from a combination of session info
        st.session_state.annotator_id = str(uuid.uuid4())[:8]
    return st.session_state.annotator_id


def save_annotation(annotation: Dict[str, Any]):
    """Save an annotation to disk."""
    annotator_id = get_annotator_id()
    timestamp = datetime.now().isoformat()
    
    annotation_record = {
        'annotator_id': annotator_id,
        'timestamp': timestamp,
        **annotation
    }
    
    # Save to annotator-specific file
    annotator_file = ANNOTATIONS_DIR / f"annotator_{annotator_id}.json"
    
    existing = []
    if annotator_file.exists():
        with open(annotator_file, 'r') as f:
            existing = json.load(f)
    
    existing.append(annotation_record)
    
    with open(annotator_file, 'w') as f:
        json.dump(existing, f, indent=2)


def get_annotation_progress() -> Dict[str, int]:
    """Get annotation progress for current annotator."""
    annotator_id = get_annotator_id()
    annotator_file = ANNOTATIONS_DIR / f"annotator_{annotator_id}.json"
    
    if annotator_file.exists():
        with open(annotator_file, 'r') as f:
            annotations = json.load(f)
        return {
            'completed': len(annotations),
            'snippet_ids': [a['snippet_id'] for a in annotations]
        }
    return {'completed': 0, 'snippet_ids': []}


# --- Streamlit UI ---

def main():
    st.set_page_config(
        page_title="Middlemarch Annotation Survey",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS for better readability
    st.markdown("""
    <style>
    .snippet-text {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        font-size: 16px;
        line-height: 1.8;
        font-family: Georgia, serif;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📚 Middlemarch Character Interaction Survey")
    st.markdown("""
    **Thank you for participating in this research study!**
    
    We're building a dataset of character interactions from George Eliot's *Middlemarch*
    to improve automated literary analysis tools.
    
    **Your task:** Read each passage and identify any character interactions you see.
    """)
    
    # Sidebar with instructions and progress
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Read the passage carefully
        2. Identify any **direct interactions** between named characters
        3. Select the characters involved
        4. Choose the type of interaction
        5. Highlight the evidence (quote from text)
        6. Submit and move to the next passage
        
        **Interaction Types:**
        - **Direct Dialogue**: Characters speaking to each other
        - **Physical Action**: Physical contact or action
        - **Observation**: One character watching another
        - **Memory/Reference**: Thinking about or mentioning another
        - **No Interaction**: No clear interaction in this passage
        """)
        
        st.divider()
        
        progress = get_annotation_progress()
        st.metric("Your Annotations", progress['completed'])
        st.caption(f"Annotator ID: {get_annotator_id()}")
    
    # Load data
    snippets = load_snippets_for_annotation()
    characters = load_characters()
    progress = get_annotation_progress()
    
    # Find next unannotated snippet
    completed_ids = set(progress['snippet_ids'])
    remaining = [s for s in snippets if s['id'] not in completed_ids]
    
    if not remaining:
        st.success("🎉 You've completed all available passages! Thank you!")
        st.balloons()
        
        # Show summary
        st.subheader("Your Contribution Summary")
        st.write(f"Total annotations: {progress['completed']}")
        return
    
    # Get current snippet
    if 'current_snippet_idx' not in st.session_state:
        st.session_state.current_snippet_idx = 0
    
    current = remaining[min(st.session_state.current_snippet_idx, len(remaining)-1)]
    
    # Progress bar
    total = len(snippets)
    completed = progress['completed']
    st.progress(completed / total, text=f"Progress: {completed}/{total} passages")
    
    # Display snippet
    st.subheader(f"Passage from {current['source']}")
    st.markdown(f"<div class='snippet-text'>{current['text']}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Annotation form
    st.subheader("Your Annotation")
    
    # Allow multiple interactions per snippet
    if 'interactions' not in st.session_state:
        st.session_state.interactions = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        char1 = st.selectbox(
            "Character 1",
            options=[""] + characters,
            help="Select the first character involved"
        )
    
    with col2:
        char2 = st.selectbox(
            "Character 2", 
            options=[""] + characters,
            help="Select the second character involved"
        )
    
    interaction_type = st.radio(
        "Interaction Type",
        options=INTERACTION_TYPES,
        horizontal=True
    )
    
    evidence = st.text_input(
        "Evidence (copy a short phrase from the text)",
        placeholder="e.g., 'she said to him'"
    )
    
    confidence = st.slider(
        "How confident are you in this annotation?",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very uncertain, 5 = Very confident"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("➕ Add Interaction", type="secondary"):
            if interaction_type == "No Interaction":
                st.session_state.interactions.append({
                    'type': 'no_interaction'
                })
                st.success("Marked as no interaction")
            elif char1 and char2 and char1 != char2:
                st.session_state.interactions.append({
                    'character_1': char1,
                    'character_2': char2,
                    'interaction_type': interaction_type,
                    'evidence': evidence,
                    'confidence': confidence
                })
                st.success(f"Added: {char1} ↔ {char2}")
            else:
                st.error("Please select two different characters")
    
    with col2:
        if st.button("🗑️ Clear All", type="secondary"):
            st.session_state.interactions = []
            st.rerun()
    
    # Show current interactions
    if st.session_state.interactions:
        st.subheader("Interactions you've identified:")
        for i, interaction in enumerate(st.session_state.interactions):
            if interaction.get('type') == 'no_interaction':
                st.write(f"{i+1}. No interaction in this passage")
            else:
                st.write(f"{i+1}. {interaction['character_1']} ↔ {interaction['character_2']} "
                        f"({interaction['interaction_type']}) - \"{interaction.get('evidence', '')}\"")
    
    st.divider()
    
    with col3:
        if st.button("✅ Submit & Next", type="primary"):
            if not st.session_state.interactions:
                st.error("Please add at least one interaction (or mark as 'No Interaction')")
            else:
                # Save annotation
                save_annotation({
                    'snippet_id': current['id'],
                    'source': current['source'],
                    'interactions': st.session_state.interactions
                })
                
                # Reset and move to next
                st.session_state.interactions = []
                st.session_state.current_snippet_idx += 1
                st.success("Annotation saved! Loading next passage...")
                st.rerun()
    
    # Skip option
    if st.button("⏭️ Skip this passage"):
        st.session_state.current_snippet_idx += 1
        st.session_state.interactions = []
        st.rerun()


if __name__ == "__main__":
    main()
