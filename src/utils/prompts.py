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
    You are an expert Knowledge Graph Engineer. Your task is to extract all entities and implied relations from the provided text.

    ### Guidelines:
     **Entities**: Identify key entities.
    - `id`: A concise, unique name (e.g., "Elon Musk" not "he").
    - `desc`: A brief keyword-rich summary (max 20 words) for context (e.g., "CEO of Tesla and SpaceX"), if the context is inadequate, use "UNKNOWN".

     **Relations**: Identify the key relation phrases implied by the query.
    - Output a list of short relation phrases (e.g., "acquired", "CEO of", "born in", "located in").
    - Do not include entity names in the relation strings.

    ### Constraints:
    - **Output Format**: Return strictly valid JSON.

    ### Schema:
    {{
    "entities": [
        {{"id": "string", "type": "string", "desc": "string"}}
    ],
    "relations": ["string"]
    }}
    Text:
    {context}

    Extract the entities and relations following the schema.  
"""

prompt_plan_first_step_str = """
    You are a decomposition assistant. Decide whether the question needs multi-hop reasoning. 
	If yes, provide the first subquestion needed to find an intermediate entity or fact. 
	If no, set subquestion to the original question.
	### Example 1:
    Question: Why did John Middleton Murry's wife die?
    JSON: {{"need_multihop": true, "subquestion": "Who is John Middleton Murry's wife?"}}

    ### Example 2:
    Question: Why did Katherine Mansfield die?
    JSON: {{"need_multihop": false, "subquestion": "Why did Katherine Mansfield die?"}}

    ### Question: 
    {query}

	### Output Format (Strictly JSON):
    {{
        "need_multihop": true/false,
        "subquestion": "string"
    }}
"""

prompt_plan_next_step_str = """
    You are an expert reasoning agent. Your goal is to answer the Original Question step by step.
	### Example:
    Which film has the director died later, Lost In The Stratosphere or Blind Man'S Eyes?
    subquestion1: Who is the director of Lost In The Stratosphere? answer1: Melville W. Brown
    subquestion2: When did Melville W. Brown die? answer2: January 31, 1938
    subquestion3: Who is the director of Blind Man'S Eyes? answer3: John Ince
    subquestion4: When did John Ince die? answer4: April 10, 1947
    final_answer: Blind Man'S Eyes.

    ### History of investigation so far:
    <history>
    {history_str}
    </history>

    ### Original Question: 
    {query}

	### INSTRUCTIONS:
	1. Look at the history. Do you have enough combined information to answer the Original Question?
	2. If YES: set "is_final" to true and provide the "final_answer".
	3. If NO: provide the next "subquestion".
	4. **CRITICAL RULES:** 
		- DO NOT ask a question you have already asked in the history.
		- If a previous result was 'UNKNOWN' or 'I don't know', DO NOT ask about it again. Try a different angle or conclude with the information you have.
	
    ### Output Format (Strictly JSON):
    {{
        "is_final": true/false,
        "subquestion": "string",
        "final_answer": "string"
    }}
	
"""

prompt_answer_with_chunks_str = """
You are an expert question-answering assistant. Answer the user's question strictly based on the provided context. Follow the three-step reasoning process below.

### Step 1: Extract Candidates
- Carefully read all context chunks.
- List every candidate answer snippet that relates to the question. Quote the exact text.
- If no relevant information exists, set step1_candidates to "None".

### Step 2: Analyze
- Compare the candidates: is there enough information to answer?
- If there are conflicts (e.g., different values, ambiguous entities), explain which to prefer and why.
- Preserve units and conditions (e.g., "$100 per seat", "50 miles per hour").
- For dates, always output absolute values (e.g., "February 20, 2022"), never relative ("Sunday").
- If information is insufficient, explain why.

### Step 3: Final Answer
- Provide the most accurate, concise answer based on the analysis above.
- If the context lacks relevant information, output "I don't know.".
- Keep it concise (typically 1-8 words), but never drop essential units or conditions.

### Context:
<context>
{context}
</context>

### Question:
{query}

### Output Format (Strictly JSON):
{{
  "step1_candidates": "List all candidate answer snippets with exact quotes from the context, or 'None'.",
  "step2_analysis": "Compare candidates, resolve conflicts, check sufficiency.",
  "final_answer": "Complete and concise answer including necessary units or absolute dates, or 'I don't know'"
}}
"""

prompt_sub_answer_lite_str = """
You are a quick search assistant. Please answer the sub-questions directly based on the context.

The output format must be JSON: {"answer": "short answer", "found": true/false}
If not mentioned in the context, set "found" to false. Do not make complex inferences or citations.
### Context:
{context}

### Sub-Question:
{query}
"""
