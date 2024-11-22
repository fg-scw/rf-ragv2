import streamlit as st
import logging
from app import init_db, process_files, generate_response, check_loaded_documents
import time
from typing import Dict, List, Optional, Union
import json

# Configuration de la page Streamlit
st.set_page_config(
    page_title="RAG Radio France",
    page_icon="📻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger for Streamlit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Style personnalisé
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .status { padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    .stAlert { padding: 1rem; margin: 1rem 0; border-radius: 0.5rem; }
    .source-info { font-size: 0.8em; color: #666; font-style: italic; margin-top: 0.5rem; }
    .confidence-high { color: #0f5132; border-left: 3px solid #0f5132; padding-left: 0.5rem; }
    .confidence-medium { color: #856404; border-left: 3px solid #856404; padding-left: 0.5rem; }
    .confidence-low { color: #721c24; border-left: 3px solid #721c24; padding-left: 0.5rem; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; background-color: #f8f9fa; }
    .source-title { font-weight: bold; color: #1e88e5; margin-bottom: 0.2rem; }
    .source-details { font-size: 0.9em; color: #666; margin-left: 1rem; border-left: 2px solid #e0e0e0; padding-left: 0.5rem; }
    .metadata-tag { display: inline-block; padding: 0.2rem 0.5rem; margin: 0.2rem; border-radius: 0.3rem; background-color: #e9ecef; font-size: 0.8em; }
    .document-section { margin-bottom: 1rem; padding: 0.5rem; border-left: 3px solid #1e88e5; background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

def parse_source_info(header: str) -> Dict[str, str]:
    """Parse les informations de source à partir de l'en-tête du document."""
    info = {
        "title": "Document sans titre",
        "source": "Source inconnue",
        "type": "Non spécifié",
        "page": "N/A"
    }
    
    try:
        if " - Source: " in header:
            source_part = header.split(" - Source: ")[1].split(" - ")[0]
            info["source"] = source_part
            info["title"] = source_part.split("/")[-1] if "/" in source_part else source_part
        if " - Type: " in header:
            info["type"] = header.split(" - Type: ")[1].split(" - ")[0]
        if " - Page: " in header:
            info["page"] = header.split(" - Page: ")[1].split("]")[0]
    except Exception as e:
        logger.error(f"Error parsing source info: {e}")
    
    return info

def display_message(role: str, content: str, metadata: Optional[Dict] = None):
    """Affiche un message avec ses métadonnées et sources."""
    with st.chat_message(role):
        st.markdown(content)
        
        if metadata and role == "assistant":
            if "sources" in metadata and metadata["sources"]:
                with st.expander("📚 Sources utilisées", expanded=True):
                    st.markdown("### Documents consultés")
                    for source in metadata["sources"]:
                        with st.container():
                            st.markdown(f"""
                            <div class="document-section">
                                <div class="source-title">{source['title']}</div>
                                <div class="source-details">
                                    <div>Source: {source['source']}</div>
                                    <div>Type: {source['type']}</div>
                                    <div>Section/Page: {source['page']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            if "confidence_score" in metadata and show_confidence:
                confidence = metadata["confidence_score"]
                confidence_class = "confidence-high" if confidence >= 0.8 else "confidence-medium" if confidence >= 0.6 else "confidence-low"
                confidence_text = "Confiance élevée" if confidence >= 0.8 else "Confiance moyenne" if confidence >= 0.6 else "Confiance faible"
                st.markdown(f'<div class="source-info {confidence_class}">{confidence_text} ({confidence:.2%})</div>', unsafe_allow_html=True)

# Cache pour l'initialisation de la base de données
@st.cache_resource
def initialize_database():
    try:
        with st.spinner("🔄 Initialisation de la base de données..."):
            logger.info("Initializing the database...")
            init_db()
            process_files()
            logger.info("Database initialized and files processed.")
            time.sleep(1)
            return True
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        return False

# Sidebar Configuration
with st.sidebar:
    st.title('🎙️ RAG Radio France')
    
    with st.expander("📊 Paramètres de recherche", expanded=False):
        k_docs = st.slider("Nombre de documents à récupérer", min_value=1, max_value=10, value=4)
        similarity_threshold = st.slider("Seuil de similarité", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    with st.expander("🛠️ Paramètres du modèle", expanded=False):
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        top_p = st.slider("Top_p", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
        max_length = st.slider("Longueur maximale", min_value=100, max_value=2000, value=1000, step=100)
    
    with st.expander("🔍 Options d'affichage", expanded=False):
        show_context = st.toggle("Afficher le contexte complet", value=False)
        show_sources = st.toggle("Afficher les sources", value=True)
        show_confidence = st.toggle("Afficher le score de confiance", value=True)
    
    with st.expander("🔄 Gestion des documents", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Rafraîchir les documents"):
                with st.spinner("🔄 Mise à jour des documents en cours..."):
                    try:
                        process_files()
                        st.success("✅ Documents mis à jour avec succès!")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la mise à jour: {str(e)}")
        
        with col2:
            if st.button("Vérifier l'état des documents"):
                with st.spinner("🔍 Vérification en cours..."):
                    doc_count, embedding_count = check_loaded_documents()
                    st.info(f"""
                    📊 État des documents:
                    - Documents traités: {doc_count}
                    - Embeddings stockés: {embedding_count}
                    """)

    
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Effacer l'historique"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Historique effacé. Comment puis-je vous aider ?"
        }]
        st.rerun()

# Initialize the database
if initialize_database():
    st.sidebar.success("✅ Base de données initialisée avec succès")
else:
    st.sidebar.error("❌ Erreur lors de l'initialisation de la base de données")
    st.stop()

# Initialiser l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Bonjour ! Je suis votre assistant Radio France. Comment puis-je vous aider aujourd'hui ?"
    }]

# Afficher l'historique des messages
for message in st.session_state.messages:
    display_message(
        message["role"],
        message["content"],
        message.get("metadata", {})
    )

# Zone de saisie utilisateur
if input_text := st.chat_input("💭 Posez votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": input_text})
    display_message("user", input_text)

    with st.status("🔍 Recherche en cours...") as status:
        st.write("Analyse des documents...")
        try:
            response, context = generate_response(input_text, temperature, top_p, max_length)
            
            sources = []
            if context:
                context_parts = context.split("\n\n")
                for part in context_parts:
                    if part.startswith("[Document"):
                        header_line = part.split("\n")[0]
                        source_info = parse_source_info(header_line)
                        sources.append(source_info)
            
            status.update(label="✅ Recherche terminée", state="complete")
            
            with st.chat_message("assistant"):
                st.markdown(response)
                if show_context and context:
                    with st.expander("📄 Contexte complet", expanded=False):
                        st.markdown(context)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "metadata": {
                    "sources": sources,
                    "confidence_score": similarity_threshold,
                    "context": context if show_context else None,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            })
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            error_message = f"❌ Désolé, une erreur s'est produite lors de la recherche: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "metadata": {
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            })
