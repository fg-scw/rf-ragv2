import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.document_loaders import S3FileLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from botocore.client import Config
import psycopg2
import os
import boto3
import logging
import tempfile
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from uuid import uuid4
import re
import json
from sklearn.decomposition import PCA
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

class DimensionalityReducer:
    def __init__(self, target_dims=1536):
        self.pca = PCA(n_components=target_dims)
        self.is_fitted = False
        self.target_dims = target_dims

    def fit_transform(self, embeddings):
        if not self.is_fitted:
            self.pca.fit(embeddings)
            self.is_fitted = True
        return self.pca.transform(embeddings)

    def transform(self, embeddings):
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet")
        return self.pca.transform(embeddings)

class CompressedEmbeddings:
    def __init__(self, base_embeddings, target_dims=1536):
        self.base_embeddings = base_embeddings
        self.target_dims = target_dims
        
    def compress_vector(self, vector):
        """Compresse un vecteur en moyennant les valeurs adjacentes."""
        if len(vector) <= self.target_dims:
            return vector
            
        # Calculer combien de valeurs originales moyenner ensemble
        compression_factor = len(vector) // self.target_dims
        remainder = len(vector) % self.target_dims
        
        compressed = []
        i = 0
        
        # Traiter les groupes complets
        while len(compressed) < self.target_dims:
            if remainder > 0:
                group_size = compression_factor + 1
                remainder -= 1
            else:
                group_size = compression_factor
                
            group = vector[i:i+group_size]
            compressed.append(sum(group) / len(group))
            i += group_size
            
        return compressed

    def embed_documents(self, texts):
        original_embeddings = self.base_embeddings.embed_documents(texts)
        return [self.compress_vector(emb) for emb in original_embeddings]

    def embed_query(self, text):
        original_embedding = self.base_embeddings.embed_query(text)
        return self.compress_vector(original_embedding)

# Configuration des embeddings
embeddings = CompressedEmbeddings(
    OpenAIEmbeddings(
        openai_api_key=os.getenv("SCW_SECRET_KEY"),
        openai_api_base=os.getenv("SCW_INFERENCE_EMBEDDINGS_ENDPOINT"),
        model="baai/bge-multilingual-gemma2",
        tiktoken_enabled=False,
    ),
    target_dims=1536
)

# Configuration de la connexion à la base de données
connection_string = f"postgresql+psycopg2://{os.getenv('SCW_DB_USER')}:{os.getenv('SCW_DB_PASSWORD')}@{os.getenv('SCW_DB_HOST')}:{os.getenv('SCW_DB_PORT')}/{os.getenv('SCW_DB_NAME')}?sslmode=require"

# Configuration S3
endpoint_s3 = f"https://s3.{os.getenv('SCW_REGION', '')}.scw.cloud"
session = boto3.session.Session()
client_s3 = session.client(
    service_name='s3',
    endpoint_url=endpoint_s3,
    aws_access_key_id=os.getenv("SCW_ACCESS_KEY", ""),
    aws_secret_access_key=os.getenv("SCW_SECRET_KEY", "")
)

# Configuration du paginator S3
paginator = client_s3.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=os.getenv("SCW_BUCKET_NAME"))

# Initialisation du vector store (sera redéfini dans init_db)
vector_store = None

def get_connection():
    """Établit une connexion à la base de données PostgreSQL."""
    try:
        conn = psycopg2.connect(
            database=os.getenv("SCW_DB_NAME"),
            user=os.getenv("SCW_DB_USER"),
            password=os.getenv("SCW_DB_PASSWORD"),
            host=os.getenv("SCW_DB_HOST"),
            port=os.getenv("SCW_DB_PORT")
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        logger.error(traceback.format_exc())
        raise

def delete_old_documents(file_key: str):
    """Supprime les anciens documents avant d'en ajouter de nouveaux."""
    try:
        # Récupérer les IDs des documents existants pour ce fichier
        docs = vector_store.similarity_search(
            query=file_key,
            k=1000,
            filter={"source": file_key}
        )

        old_ids = [doc.metadata.get("doc_id") for doc in docs if doc.metadata.get("doc_id")]

        if old_ids:
            vector_store.delete(old_ids)
            logger.info(f"Deleted {len(old_ids)} old documents for {file_key}")
    except Exception as e:
        logger.error(f"Error deleting old documents for {file_key}: {e}")
        logger.error(traceback.format_exc())

def init_db():
    """Initialise la base de données et crée les tables nécessaires."""
    global vector_store
    conn = get_connection()
    cur = conn.cursor()

    try:
        logger.info("Creating vector extension if it doesn't exist...")
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
        cur.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE;")
        cur.execute("DROP TABLE IF EXISTS object_loaded;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        logger.info("Creating tables...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS object_loaded (
                id SERIAL PRIMARY KEY,
                object_key TEXT UNIQUE,
                processed_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                file_type TEXT,
                status TEXT DEFAULT 'success',
                error_message TEXT,
                metadata JSONB DEFAULT '{}'::jsonb
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                uuid UUID PRIMARY KEY,
                name VARCHAR(50) UNIQUE,
                cmetadata JSONB
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                id TEXT PRIMARY KEY,
                collection_id UUID REFERENCES langchain_pg_collection(uuid),
                embedding vector(1536),
                document TEXT,
                cmetadata JSONB
            );
            
            CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_collection_id 
            ON langchain_pg_embedding(collection_id);
        """)
        conn.commit()

        logger.info("Initializing vector store...")
        vector_store = PGVector(
            collection_name="document_chunks",
            connection=connection_string,
            embeddings=embeddings,
            collection_metadata={"description": "Document embeddings collection"}
        )
        
        logger.info("Vector store initialized successfully")

    except Exception as e:
        logger.error(f"An error occurred during database initialization: {e}")
        logger.error(traceback.format_exc())
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def check_vector_store():
    """Vérifie l'état du vector store avec une requête test."""
    try:
        if not vector_store:
            logger.error("Vector store not initialized")
            return

        test_query = "test"
        logger.info(f"Testing vector store with query: {test_query}")

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 2,
                "fetch_k": 4,
                "lambda_mult": 0.7
            }
        )

        docs = retriever.invoke(test_query)

        for i, doc in enumerate(docs):
            logger.info(f"Document {i+1} source: {doc.metadata.get('source')}")
            logger.info(f"Document {i+1} content length: {len(doc.page_content)}")
            logger.info(f"Document {i+1} preview: {doc.page_content[:200]}...")
            logger.info("---")

    except Exception as e:
        logger.error(f"Error checking vector store: {e}")
        logger.error(traceback.format_exc())

# Configuration optimisée du text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n\n", "\n\n", "\n", ".", "!", "?", ";", ":", ",", " ", ""],
    chunk_size=512,
    chunk_overlap=102,
    length_function=len,
    keep_separator=True,
    add_start_index=True,
    strip_whitespace=True,
    is_separator_regex=False
)

def clean_text(text: str) -> str:
    """Nettoie et normalise le texte avant la tokenization."""
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[''′]', "'", text)
    text = re.sub(r'[‐‑‒–—―]', "-", text)
    text = re.sub(r'[""„‟]', '"', text)
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)
    text = re.sub(r'^\s*[•●○▪︎■]\s*', '- ', text, flags=re.MULTILINE)
    return text.strip()

def detect_document_structure(text: str) -> Dict[str, Any]:
    """Analyse la structure du document et détecte les patterns de contenu."""
    structure = {
        "has_headers": bool(re.search(r'^[A-Z][^.!?]*[\n\r]', text, re.MULTILINE)),
        "has_lists": bool(re.search(r'^\s*[-•*]\s', text, re.MULTILINE)),
        "has_numbers": bool(re.search(r'^\s*\d+\.?\s', text, re.MULTILINE)),
        "has_tables": bool(re.search(r'[|┃]\s*\w+\s*[|┃]', text)),
        "avg_sentence_length": len(re.findall(r'[.!?]+', text)) / max(len(text.split('\n')), 1),
        "avg_paragraph_length": len(text.split('\n\n')),
        "total_length": len(text),
        "language_hint": "fr" if re.search(r'\b(le|la|les|un|une|des)\b', text.lower()) else "en"
    }
    structure["complexity"] = "high" if structure["avg_sentence_length"] > 20 else "medium" if structure["avg_sentence_length"] > 10 else "low"
    return structure

def process_document(doc: Document, file_key: str, metadata: dict) -> Tuple[List[Document], List[str]]:
    """Traite un document avec chunking amélioré et métadonnées enrichies."""
    cleaned_content = clean_text(doc.page_content)
    structure = detect_document_structure(cleaned_content)

    local_splitter = text_splitter
    if structure["has_headers"] or structure["has_lists"]:
        local_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
            chunk_size=350,
            chunk_overlap=35,
            keep_separator=True
        )

    chunks = local_splitter.split_text(cleaned_content)
    processed_docs = []
    doc_ids = []

    for i, chunk in enumerate(chunks):
        doc_id = f"{file_key}-{metadata.get('page_number', 0)}-{i}-{str(uuid4())[:8]}"
        
        content_type = "unknown"
        if re.match(r'^#+ ', chunk):
            content_type = "header"
        elif re.match(r'^\d+\. ', chunk):
            content_type = "numbered_list"
        elif re.match(r'^[-•*] ', chunk):
            content_type = "bullet_list"
        elif len(chunk.split('.')) > 3:
            content_type = "paragraph"
        elif structure["has_tables"] and re.search(r'[|┃]', chunk):
            content_type = "table"

        chunk_metadata = {
            **metadata,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk),
            "content_type": content_type,
            "source": file_key,
            "document_structure": structure,
            "word_count": len(chunk.split()),
            "sentence_count": len(re.split(r'[.!?]+', chunk)),
            "processing_timestamp": datetime.now().isoformat(),
            "doc_id": doc_id
        }

        processed_docs.append(
            Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
        )
        doc_ids.append(doc_id)

    return processed_docs, doc_ids

def process_files():
    """Traite les fichiers du bucket S3, les découpe en chunks et les stocke dans la base vectorielle."""
    if not vector_store:
        logger.error

def process_files():
    """Traite les fichiers du bucket S3, les découpe en chunks et les stocke dans la base vectorielle."""
    if not vector_store:
        logger.error("Vector store not initialized")
        return

    conn = get_connection()
    cur = conn.cursor()

    try:
        for page in page_iterator:
            for obj in page.get('Contents', []):
                file_key = obj['Key']
                logger.info(f"Processing file from S3: {file_key}")

                file_extension = file_key.lower().split('.')[-1]
                logger.info(f"File type: {file_extension}")

                cur.execute("SELECT object_key, processed_date FROM object_loaded WHERE object_key = %s", (file_key,))
                response = cur.fetchone()

                # Traiter seulement les nouveaux fichiers ou les fichiers modifiés
                if response is None or obj['LastModified'].replace(tzinfo=None) > response[1].replace(tzinfo=None):
                    try:
                        # Supprimer les anciens documents si le fichier existe déjà
                        if response is not None:
                            delete_old_documents(file_key)

                        # Téléchargement du fichier
                        with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as temp_file:
                            client_s3.download_fileobj(
                                Bucket=os.getenv("SCW_BUCKET_NAME"),
                                Key=file_key,
                                Fileobj=temp_file
                            )
                            temp_file_path = temp_file.name
                            logger.info(f"Downloaded to: {temp_file_path}")

                        try:
                            # Sélection du loader approprié
                            if file_extension == 'pdf':
                                loader = PyPDFLoader(temp_file_path)
                                logger.info("Using PyPDFLoader")
                            elif file_extension in ['doc', 'docx']:
                                loader = UnstructuredWordDocumentLoader(temp_file_path)
                                logger.info("Using UnstructuredWordDocumentLoader")
                            else:
                                loader = S3FileLoader(
                                    bucket=os.getenv("SCW_BUCKET_NAME"),
                                    key=file_key,
                                    endpoint_url=endpoint_s3,
                                    aws_access_key_id=os.getenv("SCW_ACCESS_KEY", ""),
                                    aws_secret_access_key=os.getenv("SCW_SECRET_KEY", "")
                                )
                                logger.info("Using S3FileLoader")

                            # Chargement et traitement du document
                            documents = loader.load()
                            logger.info(f"Loaded {len(documents)} documents")

                            total_chunks = 0
                            for doc_idx, doc in enumerate(documents):
                                base_metadata = {
                                    "file_type": file_extension,
                                    "page_number": doc_idx + 1,
                                    "timestamp": obj['LastModified'].isoformat(),
                                    "file_name": file_key,
                                    "processing_batch": datetime.now().isoformat(),
                                    "file_size": obj.get('Size', 0)
                                }

                                # Traitement du document avec IDs
                                processed_chunks, chunk_ids = process_document(doc, file_key, base_metadata)
                                logger.info(f"Document {doc_idx + 1}: Split into {len(processed_chunks)} chunks")
                                total_chunks += len(processed_chunks)

                                # Traitement par lots de 10 chunks
                                for i in range(0, len(processed_chunks), 10):
                                    batch_docs = processed_chunks[i:i + 10]
                                    batch_ids = chunk_ids[i:i + 10]
                                    try:
                                        vector_store.add_documents(
                                            documents=batch_docs,
                                            ids=batch_ids
                                        )
                                        logger.info(f"Added batch of chunks {i}-{i + len(batch_docs)} to vector store")
                                    except Exception as e:
                                        logger.error(f"Error adding batch {i}-{i + len(batch_docs)}: {e}")
                                        raise

                            # Mise à jour de la base de données
                            processed_metadata = {
                                "total_documents": len(documents),
                                "total_chunks": total_chunks,
                                "processing_timestamp": datetime.now().isoformat(),
                                "file_size": obj.get('Size', 0),
                                "file_type": file_extension
                            }

                            if response is None:
                                cur.execute(
                                    """
                                    INSERT INTO object_loaded
                                    (object_key, processed_date, file_type, status, metadata)
                                    VALUES (%s, %s, %s, 'success', %s)
                                    """,
                                    (file_key, obj['LastModified'], file_extension, json.dumps(processed_metadata))
                                )
                            else:
                                cur.execute(
                                    """
                                    UPDATE object_loaded
                                    SET processed_date = %s,
                                        file_type = %s,
                                        status = 'success',
                                        error_message = NULL,
                                        metadata = %s
                                    WHERE object_key = %s
                                    """,
                                    (obj['LastModified'], file_extension, json.dumps(processed_metadata), file_key)
                                )
                            conn.commit()
                            logger.info(f"Successfully processed file: {file_key}")

                        finally:
                            # Nettoyage du fichier temporaire
                            os.unlink(temp_file_path)
                            logger.info("Temporary file cleaned up")

                    except Exception as e:
                        logger.error(f"Error processing file {file_key}: {e}")
                        logger.error(traceback.format_exc())

                        error_metadata = {
                            "error_timestamp": datetime.now().isoformat(),
                            "error_type": type(e).__name__,
                            "file_size": obj.get('Size', 0),
                            "file_type": file_extension
                        }

                        error_message = str(e)[:500]
                        if response is None:
                            cur.execute(
                                """
                                INSERT INTO object_loaded
                                (object_key, processed_date, file_type, status, error_message, metadata)
                                VALUES (%s, %s, %s, 'error', %s, %s)
                                """,
                                (file_key, obj['LastModified'], file_extension, error_message, json.dumps(error_metadata))
                            )
                        else:
                            cur.execute(
                                """
                                UPDATE object_loaded
                                SET processed_date = %s,
                                    file_type = %s,
                                    status = 'error',
                                    error_message = %s,
                                    metadata = %s
                                WHERE object_key = %s
                                """,
                                (obj['LastModified'], file_extension, error_message, json.dumps(error_metadata), file_key)
                            )
                        conn.commit()

    except Exception as e:
        logger.error(f"Error during file processing: {e}")
        logger.error(traceback.format_exc())
        conn.rollback()
    finally:
        cur.close()
        conn.close()

# Configuration du LLM
llm = ChatOpenAI(
    base_url=os.getenv("SCW_INFERENCE_DEPLOYMENT_ENDPOINT"),
    api_key=os.getenv("SCW_SECRET_KEY"),
    model="llama-3.1-70b-instruct",
    temperature=0.7
)

# Template de prompt amélioré
prompt_template = """Vous êtes un expert en analyse de documents. Utilisez les informations du contexte fourni pour répondre à la question de manière précise et structurée.

Contexte fourni :
{context}

Question posée : {question}

Instructions pour la réponse :
1. Analysez attentivement le contexte fourni
2. Identifiez les informations pertinentes et leur source
3. Structurez votre réponse de manière claire et logique
4. Citez des passages spécifiques du contexte quand c'est pertinent
5. Si le contexte ne contient pas assez d'informations, indiquez-le clairement
6. Si plusieurs sources se contredisent, mentionnez les différentes perspectives
7. Évitez les suppositions non fondées sur le contexte

Format de réponse souhaité :
- Commencez par une réponse directe à la question
- Développez avec les détails pertinents
- Citez les sources spécifiques quand nécessaire
- Concluez avec un bref résumé si la réponse est longue

Réponse :"""

custom_rag_prompt = PromptTemplate.from_template(prompt_template)
custom_rag_chain = create_stuff_documents_chain(llm, custom_rag_prompt)

def generate_response(input: str, temperature: float, top_p: float, max_length: int) -> tuple[str, str]:
    """
    Génère une réponse basée sur le contenu vectorisé et les paramètres fournis.
    """
    try:
        if not vector_store:
            return "Le système n'est pas correctement initialisé. Veuillez réessayer plus tard.", ""

        cleaned_input = input.strip()
        logger.info(f"Processing query: {cleaned_input}")

        # Augmentation du nombre de documents récupérés
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,  # Augmenté de 4 à 8
                "fetch_k": 16,  # Augmenté de 8 à 16
                "lambda_mult": 0.7,
                "filter": None  # Assurez-vous qu'aucun filtre n'est appliqué
            }
        )

        docs = retriever.invoke(cleaned_input)
        logger.info(f"Found {len(docs)} relevant documents")
        
        for i, doc in enumerate(docs):
            logger.info(f"Document {i+1}: Metadata - {doc.metadata}, Preview - {doc.page_content[:200]}")

        if docs:
            # Préparation du contexte avec plus d'informations
            context_text = "\n\n".join([
                f"[Document {i+1} - Source: {doc.metadata.get('source', 'Source inconnue')} - "
                f"Type: {doc.metadata.get('content_type', 'Non spécifié')} - "
                f"Page: {doc.metadata.get('page_number', 'N/A')}]\n"
                f"{doc.page_content}"
                for i, doc in enumerate(docs)
            ])

            logger.info(f"Context length: {len(context_text)} characters")
            logger.info(f"Context Text: {context_text}")

            # Mise à jour des paramètres du LLM
            llm.temperature = temperature
            llm.top_p = top_p
            llm.max_tokens = max_length

            # **Ajoutez le log ici :**
            logger.info(f"Prompt sent to LLM: {custom_rag_prompt.format(context=context_text, question=cleaned_input)}")

            # Génération de la réponse
            response = custom_rag_chain.invoke({
                "question": cleaned_input,
                "context": docs
            })

            return response, context_text
        else:
            logger.warning(f"No relevant documents found for query: {cleaned_input}")
            return ("Je ne trouve pas d'information pertinente dans ma base de connaissances. "
                   "Veuillez vérifier que les documents ont été correctement chargés."), ""

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        logger.error(traceback.format_exc())
        return ("Une erreur s'est produite lors du traitement de votre demande. "
                "Veuillez vérifier les logs pour plus de détails."), ""

# Ajout d'une fonction de vérification des documents chargés
def check_loaded_documents():
    """Vérifie les documents chargés dans le vector store."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Vérification des documents traités
        cur.execute("SELECT COUNT(*) FROM object_loaded WHERE status = 'success'")
        doc_count = cur.fetchone()[0]
        
        # Vérification des embeddings
        cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
        embedding_count = cur.fetchone()[0]
        
        logger.info(f"Documents successfully loaded: {doc_count}")
        logger.info(f"Total embeddings stored: {embedding_count}")
        
        return doc_count, embedding_count
        
    except Exception as e:
        logger.error(f"Error checking loaded documents: {e}")
        return 0, 0
    finally:
        cur.close()
        conn.close()
