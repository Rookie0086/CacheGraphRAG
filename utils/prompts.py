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

# prompt_answer_with_chunks_str = """
# You are an expert question-answering assistant. Your task is to answer the user's question based strictly on the provided context.

# ### Instructions:
# 1. Carefully read all the provided context chunk. The "ts" field represents the latest update time of the chunk, you should use it as an important basis for judging the credibility of the text.
# 2. The exact phrasing of the answer might not be explicitly stated. You are allowed to make logical deductions and synthesize information ONLY if the evidence is strongly supported by the context.
# 3. If the context completely lacks relevant information to answer the question, output "I don't know." in the "final_answer" field.
# 4. Provide your thought process in the "reasoning" field first, then provide a highly concise answer (1-5 words if possible) in the "final_answer" field.

# ### Context:
# {context}

# ### Question:
# {query}

# ### Output Format (Strictly JSON):
# {{
#   "reasoning": "Briefly explain how you found the answer from the context.",
#   "final_answer": "Concise answer here"
# }}
# """

# prompt_answer_with_chunks_str = """
# You are an expert question-answering assistant. Your task is to answer the user's question based strictly on the provided context.

# ### Instructions:
# 1. Carefully read all the provided context chunks. The "ts" field represents the latest update time of the chunk, you should use it as an important basis for judging the credibility of the text.
# 2. The exact phrasing of the answer might not be explicitly stated. You are allowed to make logical deductions and synthesize information ONLY if the evidence is strongly supported by the context.
# 3. If the context completely lacks relevant information to answer the question, output "I don't know." in the "final_answer" field.
# 4. **Preserve Units and Conditions:** Never drop essential units, metrics, or qualifying conditions. If the text specifies a condition alongside a value (e.g., "$100 per seat", "50 miles per hour"), you MUST include the complete phrase in your final answer.
# 5. Provide your thought process in the "reasoning" field first. Then, provide the final answer in the "final_answer" field. Keep it concise (typically 1-8 words), but ALWAYS prioritize completeness and accuracy over strict brevity.

# ### Context:
# <context>
# {context}
# </context>

# ### Question:
# {query}

# ### Output Format (Strictly JSON):
# {{
#   "reasoning": "Briefly explain how you found the answer from the context.",
#   "final_answer": "Complete and concise answer here, including necessary units or full names."
# }}

# """

prompt_answer_with_chunks_str = """
You are an expert question-answering assistant. Your task is to answer the user's question based strictly on the provided context.

### Instructions:
1. Carefully read all the provided context chunks. The "ts" field represents the latest update time of the chunk, you should use it as an important basis for judging the credibility of the text.
2. The exact phrasing of the answer might not be explicitly stated. You are allowed to make logical deductions and synthesize information ONLY if the evidence is strongly supported by the context.
3. If the context completely lacks relevant information to answer the question, output "I don't know." in the "final_answer" field.
4. **Preserve Units and Conditions:** Never drop essential units, metrics, or qualifying conditions. If the text specifies a condition alongside a value (e.g., "$100 per seat", "50 miles per hour"), you MUST include the complete phrase in your final answer.
5. **Absolute Temporal and Data Specificity:** - When asked "When" or about a date, ALWAYS extract the exact, absolute date (e.g., "February 20, 2022") rather than relative days (e.g., "Sunday" or "tomorrow").
   - When asked for a statistic, if the context contains multiple conflicting values (e.g., "seasonally adjusted 3.7%" vs "not seasonally adjusted 3.8%"), prioritize the standard headline metric (seasonally adjusted) or the value most consistently supported across multiple chunks.
6. **Gather Before Concluding:** In the "reasoning" field, you MUST explicitly list all candidate answers found in the text before making your final decision. Compare them, then provide the most accurate final answer. Keep the "final_answer" concise (typically 1-8 words), prioritizing completeness and accuracy.

### Context:
<context>
{context}
</context>

### Question:
{query}

### Output Format (Strictly JSON):
{{
  "reasoning": "Step 1: Gather all candidates from the text. Step 2: Compare them based on instructions. Step 3: Conclude the best answer.",
  "final_answer": "Complete and concise answer here, including necessary units or absolute dates."
}}
"""