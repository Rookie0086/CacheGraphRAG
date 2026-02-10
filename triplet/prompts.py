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
    ## 1. Overview
    You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. Try to capture as much information from the text as possible without sacrificing accuracy.
    Do not add any information that is not explicitly mentioned in the text. The text document will only be provided to you ONCE. After reading it, both you and we will no longer have access to it (like a closed-book exam).
    Therefore, extract all self-contained information needed to reconstruct the knowledge. Do NOT use vague pronouns like "this", "that", or "it" to refer to prior context in the text. Always use full, explicit names or phrases that can stand alone.
    - **Nodes** represent entities and concepts.
    - The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible to a vast audience.
    ## 2. Labeling Nodes
    - **Consistency**: Ensure you use available types for node labels. Ensure you use basic or elementary types for node labels.
    - For example, when you identify an entity representing a person, always label it as **'person'**. Avoid using more specific terms like 'mathematician' or 'scientist'.
    - **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
    - **Relationships** represent connections between entities or concepts. Ensure consistency and generality in relationship types when constructing knowledge graphs. Instead of using specific and momentary type such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. Make sure to use general and timeless relationship types!
    ## 3. Coreference Resolution
    - **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency. If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
    always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID. Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
    ## 4. Strict Compliance
    Adhere to the rules strictly. Non-compliance will result in termination.
    
    -Goal-
    Given a text document, identify all entities from the text and all relationships among the identified entities.
    
    -Steps-
    1. Identify all entities. For each identified entity, extract its type, name, description, and properties.
    - type: One of the following types, but not limited to: ["Person", "Movie", "Tv", "Award", "Geo", "Genre", "Year", "Organization", "Event"]. Please refrain from creating a new entity type, always try to fit the entity to one of the provided types first.
    - name: Name of the entity, use the same language as input text. If English, capitalize the name.
    - description: Comprehensive and general description (under 50 words) of the entity.
    - properties: Entity properties are key-value pairs modeling special relations where an entity has **only one valid value at any point in its lifetime**. These properties **do not change frequently**.
      - Each type of entity can have a distinct set of properties.
      - If any properties were not mentioned in the text, please skip them.
      - Only include those properties with a **valid value**.
      - Example entity properties: A person-typed entity may have a birthday and nationality. A movie-typed entity may have a release date and language. What they have in common is that they tend to have one valid value at any point in their lifetime.
    Format each entity as a list of 3 string elements and a set of key-value pairs: \
    ["type", "name", "description", {{"key": "val", ...}}], assign this list to a key named "ent_i", where i is the entity index.
    
    2. Among the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other and extract their description and potential properties.
    - source_entity_name: name of the source entity, *MUST BE* one of the entity names identified in step 1 (the "name").
    - relation_name: up to *three words* as a predicate describing the general relationship between the source entity and target entity, capitalized and joined with underscores .
    - target_entity_name: name of the target entity, *MUST BE* one of the entity names identified in step 1 (the "name").
    - description: short and concise explanation as to why you think the source entity and the target entity are related to each other
    - relation_properties: Relation properties are special complement parts of relations, they store information that is not manifest by the relation name alone. 
        - Each type of relation can have a distinct set of properties.
        - Example relation properties: A WORK_IN relation may have an occupation. A HAS_POPULATION relation may have the value of the population.
    Format each relationship as a list of 4 string elements and a set of key-value pairs: \
    ["source_entity_name", "relation_name", "target_entity_name", "description", {{"key": "val", ...}}], assign this list to a key named "rel_i", where i is the relation index.
    
    To better extract relations, please follow these two sub-steps exactly.
    a. Identify **exclusive relations that evolve over time** (time-sensitive exclusivity). These relationships should be extracted as **temporal relations** instead of properties.
    - If a relationship **can change over time but only one value is valid at any given moment**, it must be modeled as a **temporal relationship with timestamps**. Example relationships include:
     - A person works at only one company at a time: (Person: JOHN)-[WORKS_AT, props: {{valid_from: 2019-01-01, valid_until: 2021-06-01}}]->(Company: IBM).
     - A person resides in only one place at a time: (Person: LISA)-[LIVES_IN, props: {{valid_from: 2021-03-14, valid_until: None}}]->(Geo: BOSTON).
     - A geographic region has a population that changes over time: (Geo: UNITED STATES)-[HAS_POPULATION, props: {{valid_from: 2025, valid_until: None, population: 340.1 million}}]->(Geo: UNITED STATES).
    - These relationships should be formatted as a list of 4 string elements and a set of key-value pairs: ["source_entity", "relation_name", "target_entity", "relation_description", {{"valid_from": "YYYY-MM-DD", "valid_until": "YYYY-MM-DD", "key": "val", ...}}].
    
    b. Identify **accumulative relations** (non-exclusive relationships). These relations **do not need deprecation** and can have multiple values coexisting. Example relationships include:
    - Actors can act in multiple movies: (Person: AMY)-[ACTED_IN, props: {{character: Anna, year: 2019}}]->(Movie: A GOOD MOVIE).
    - A person can have multiple skills: (Person: AMY)-[HAS_SKILL, props: {{skill: jogging}}]->(Person: AMY).
    - A person can have multiple friends: (Person: JENNY)-[HAS_FRIEND]->(Person: AMY).
    - Format these relations as: ["source_entity", "relation_name", "target_entity", "relation_description", {{"key": "val", ...}}].
    
    3. Return output as a flat JSON. *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*  
    **You must attempt to extract as many entities and relations as you can.** It’s fine to infer entity roles and connections when strongly suggested by context or scene description.
    But it's crucial that "source_entity_name" and "target_entity_name" in the identified relations, *MUST BE* one of the identified entity names. 

    Here are some examples:
    "examples": [
        {{
            "text": "Marie Curie, a Polish-French physicist, was born in Warsaw and later became a professor at the University of Paris. She was awarded two Nobel Prizes: one in Physics (1903) and one in Chemistry (1911).",
            "output": {{
                "ent_0": ["Person", "Marie Curie", "Polish-French physicist", {{"nationality": "Polish-French", "birth_place": "Warsaw"}}],
                "ent_1": ["Geo", "Warsaw", "Marie Curie's birthplace", {{}}],
                "ent_2": ["Organization", "University of Paris", "Where Marie Curie was a professor", {{}}],
                "ent_3": ["Award", "Nobel Prize in Physics", "Award won by Marie Curie in 1903", {{"year": "1903", "type": "Nobel Prize"}}],
                "ent_4": ["Award", "Nobel Prize in Chemistry", "Award won by Marie Curie in 1911", {{"year": "1911", "type": "Nobel Prize"}}],
                "rel_0": ["Marie Curie", "WORKS_AT", "University of Paris", "Marie Curie worked at the University of Paris", {{"valid_from": "None", "valid_until": "None", "occupation": "professor"}}],
                "rel_1": ["Marie Curie", "WON", "Nobel Prize in Physics", "Marie Curie won the Nobel Prize in Physics in 1903", {{"year": "1903"}}],
                "rel_2": ["Marie Curie", "WON", "Nobel Prize in Chemistry", "Marie Curie won the Nobel Prize in Chemistry in 1911", {{"year": "1911"}}]
            }},
            "explanation": "The nationality and birth place is life-time exclusive, so they are Marie's properties. WORKS_AT is modeled as exclusive relations, given the valid period. WON award is an accumulative relationship, so no valid period is given."
        }},
        {{
            "text": "Inception, a science fiction film released in 2010 and starring Leonardo DiCaprio, was lauded for its groundbreaking visual effects. At the 83rd Academy Awards (2010), Inception received the Oscar for Best Visual Effects. Leonardo DiCaprio, born on November 11, 1974, portrayed the lead character in this visually stunning film with a production budget of $160 million.",
            "output": {{
                "ent_0": ["Movie", "Inception", "A science fiction film released in 2010", {{"release_year": 2010, "budget": 160000000}}],
                "ent_1": ["Person", "Leonardo DiCaprio", "American actor and Hollywood A-lister", {{"birthday": "November 11, 1974"}}],
                "ent_2": ["Award", "Visual Effects", "An award for the best visual effects", {{"year": 2011, "ceremony_number": 83, "type": "OSCAR"}}],
                "rel_0": ["Leonardo DiCaprio", "ACTED_IN", "Inception", "Leonardo DiCaprio acted in the film Inception", {{}}],
                "rel_1": ["Inception", "WON", "Visual Effects", "Inception won the Best Visual Effects Oscar at the 83rd Academy Awards", {{"winner": "true", "movie": "Inception"}}],
            }},
            "explanation": "Although the text refers to the “83rd Academy Awards (2010)”, the extraction uses 2011 as the year property of the award entity, since the actual 83rd award event took place in 2011, consistent with the domain-specific hint that the Oscars are typically held one year after the movie’s release."
        }}
    ],

    Text:
    {context}

    Output format (flat JSON):
    {{
      "ent_i": ["type", "name", "description", {{"key": "val", ...}}],
      "rel_j": ["source_entity_name", "relation_name", "target_entity_name", "relation_description", {{"key": "val", ...}}],
      ...
    }}
    **REMINDER**: You are rewarded for high coverage and precise reasoning. Extract as much useful information as you can.    
    
"""
