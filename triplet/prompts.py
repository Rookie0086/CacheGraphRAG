alignment_prompt = """
#### Task

Perform **Entity Alignment** and **Relation Alignment** across all collections of triplets provided in the nested list. Ensure that identical entities and relationships are standardized consistently across the entire input, not just within individual collections.

- **Entity Alignment**: Identify and merge different expressions of the same entity. Ensure all variations of an entity are unified into a single, standardized name.
- **Relation Alignment**: Identify and merge different expressions of the same relationship. Ensure all variations of a relationship are unified into a single, standardized form.

**Input Format**:
The input is a nested list, where each sub-list is a collection of triplets. Each triplet follows this format:
`["Subject Entity", "Relation", "Object Entity"]`

Example **Input**:
[
    [
        ["International Business Machines Corporation",
            "headquartered_in", "New York"],
        ["IBM", "based_in", "New York"],
        ["International Business Machines Corporation", "owns", "Red Hat"]
    ],
    [
        ["IBM", "acquired", "Red Hat"],
        ["IBM", "has_location", "Armonk"]
    ]
]


**Expected Output Format**:
- Each collection has aligned entities and relationships, ensuring consistency across all collections.
- All variations of the same **entity** are unified into a single standardized name across all collections.
- All variations of the same **relationship** are unified into a single standardized form across all collections.
- **The number of triplets in the output must match the number in the input, and the order of triplets in the output must align exactly with the order in the input.**
- Ensure that the output is **valid JSON** format.

Example **Output**:
```json
{{
"aligned_triplets": [
    [
        ["IBM", "headquartered_in", "New York"],
        ["IBM", "headquartered_in", "New York"],
        ["IBM", "owns", "Red Hat"]
    ],
    [
        ["IBM", "owns", "Red Hat"],
        ["IBM", "has_location", "Armonk"]
    ]
]
}}
```


### Task Instructions:
1. Identify different expressions for the same entity (e.g., "IBM" and "International Business Machines Corporation") across all collections, and standardize them to one name.
2. Identify different expressions of the same relationship (e.g., "headquartered_in" and "based_in") across all collections, and standardize them to one form.
3. Return the aligned result as a **valid JSON object** with the key `aligned_triplets` containing the aligned triplets, ensuring entities and relationships are consistent across all collections
4. **The output must exactly match the input in terms of the number and order of triplets**. Provide only the final aligned result—no explanation or analysis.

**Input**:
{input_data}

**Output**:
"""


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