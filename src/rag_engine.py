"""
rag_engine.py
RAG (Retrieval-Augmented Generation) engine for threat intelligence.
Indexes MITRE ATT&CK-style descriptions and CVE notes, then answers
questions about detected attack types using Gemini + retrieved context.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")

# Seed knowledge base — expand this with real MITRE ATT&CK / CVE entries
THREAT_KB = [
    {
        "id": "ddos_001",
        "text": "DDoS (Distributed Denial of Service) attacks flood a target with "
                "traffic from multiple sources to exhaust bandwidth or resources. "
                "Mapped to MITRE ATT&CK T1498. Mitigation: rate limiting, traffic "
                "scrubbing, upstream ISP filtering.",
    },
    {
        "id": "portscan_001",
        "text": "Port scanning is a reconnaissance technique (MITRE ATT&CK T1046) "
                "used to discover open ports and running services on a target host, "
                "often preceding an exploitation attempt. Mitigation: firewall rules, "
                "intrusion detection alerts on sequential connection attempts.",
    },
    {
        "id": "bruteforce_001",
        "text": "Brute force attacks (MITRE ATT&CK T1110) attempt many username/password "
                "combinations to gain unauthorized access. Mitigation: account lockout "
                "policies, rate limiting, multi-factor authentication.",
    },
    {
        "id": "botnet_001",
        "text": "Botnet traffic involves compromised hosts communicating with a "
                "command-and-control (C2) server (MITRE ATT&CK T1071). Indicators "
                "include periodic beaconing and connections to known-bad IPs.",
    },
    {
        "id": "infiltration_001",
        "text": "Infiltration attacks involve an attacker gaining an initial foothold, "
                "often via a malicious payload or exploit (MITRE ATT&CK TA0001 - "
                "Initial Access), followed by lateral movement within the network.",
    },
]


def build_knowledge_base():
    """Create (or reload) a persistent Chroma collection of threat intel docs."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="threat_intel", embedding_function=embed_fn
    )

    existing_ids = set(collection.get()["ids"])
    new_entries = [e for e in THREAT_KB if e["id"] not in existing_ids]

    if new_entries:
        collection.add(
            ids=[e["id"] for e in new_entries],
            documents=[e["text"] for e in new_entries],
        )
    return collection


def retrieve_context(collection, query, n_results=2):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if results["documents"] else []


def ask_threat_intel(query, api_key, model_name="gemini-3.6-flash"):
    """
    Full RAG call: retrieve relevant threat-intel context, then ask Gemini
    to answer the query grounded in that context.
    """
    import google.generativeai as genai

    collection = build_knowledge_base()
    context_docs = retrieve_context(collection, query)
    context_text = "\n".join(context_docs)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = f"""You are a SOC (Security Operations Center) analyst assistant.
    Use the following threat intelligence context to answer the question.
    If the context doesn't cover it, say so honestly.

    Context:
    {context_text}

    Question: {query}
    """

    response = model.generate_content(prompt)
    return response.text, context_docs


if __name__ == "__main__":
    collection = build_knowledge_base()
    print("Knowledge base loaded with", collection.count(), "entries")
