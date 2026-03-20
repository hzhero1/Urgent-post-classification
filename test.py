prompt = """You are a classifier that predicts the urgency level of posts in a MOOC discussion forum.

Input: a single post as a string of text.  
Output: a single integer from 1 to 7. Output only the integer.

Urgency levels:
1: No reason to read the post  
2: Not actionable, read if time  
3: Not actionable, may be interesting  
4: Neutral, respond if spare time  
5: Somewhat urgent, good idea to reply, a teaching assistant might suffice  
6: Very urgent: good idea for the instructor to reply  
7: Extremely urgent: instructor definitely needs to reply

Choose the level that best matches the urgency implied by the post."""

print(prompt)