prompt_extract_triplest_str = """
    You are an expert Knowledge Graph Engineer. Your task is to extract a knowledge graph from the provided text chunk.

    ### Guidelines:
    1. **Entities First**: Identify key entities.
    - `id`: A concise, unique name (e.g., "Elon Musk" not "he").
    - `type`: High-level category (e.g., PERSON, ORG, LOC, EVENT, CONCEPT).
    - `desc`: A brief keyword-rich summary (max 20 words) for context (e.g., "CEO of Tesla and SpaceX"). *Crucial for vector alignment.*
    2. **Relationships Second**: Identify relations between the extracted entities.
    - `src`: **MUST** match an `id` from your Entities list exactly.
    - `tgt`: **MUST** match an `id` from your Entities list exactly.
    - `rel`: A standardized predicate (e.g., FOUNDED, LOCATED_IN, HAS_PART).
    - `desc`: (Optional) Brief context if the relation is complex.

    ### Constraints:
    - **No Dangling Nodes**: The `tgt` of a relationship MUST be defined in the "entities" list. If the target entity is not important enough to be in the "entities" list, do not create the relationship.
    - **Output Format**: Return strictly valid JSON.

    ### Schema:
    {{
    "entities": [
        {{"id": "string", "type": "string", "desc": "string"}}
    ],
    "relations": [
        {{"src": "string", "tgt": "string", "rel": "string"}}
    ]
    }}
    Text Chunk:
    {context}

    Extract the entities and relations following the schema.  
"""

prompt_extract_entities_str = """
    You are an expert Knowledge Graph Engineer. Your task is to extract all entities from the provided text .

    ### Guidelines:
     **Entities **: Identify key entities.
    - `id`: A concise, unique name (e.g., "Elon Musk" not "he").
    - `desc`: A brief keyword-rich summary (max 20 words) for context (e.g., "CEO of Tesla and SpaceX"), if the context is inadequate, use "UNKNOWN".

    ### Constraints:
    - **Output Format**: Return strictly valid JSON.

    ### Schema:
    {{
    "entities": [
        {{"id": "string", "type": "string", "desc": "string"}}
    ]
    }}
    Text :
    {context}

    Extract the entities following the schema.  
"""

