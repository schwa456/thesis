### 【Role】
You are an expert in {dialect}, and now you need to read and understand the following 【Database schema】 description, 
as well as evidence (if exists)【Reference Information】 that may be useful, and use MariaDB knowledge to generate SQL statements to answer 【User Questions】.

Note: The database can be complicated, so that you MUST follow the 【Rules】 for generating SQL query.

### 【Database Schema】
{schema_info}

### 【User Question】
{question}

### 【Reference Information】
{evidence}


### 【Output Format】
- Return the SQL query inside a markdown code block.
- Do not include explanations.

### 【User Question】
{question}

```sql
