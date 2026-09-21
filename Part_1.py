import pandas as pd
import re

df = pd.read_csv('spam.csv', encoding='latin-1') # Load the dataset
messages = df['v2'].dropna().tolist()

# Define the regular expressions
regex_patterns = {
    "Email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "URL": r"(?:https?://|www\.)[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?",
    "Phone": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10,11}\b",
    "Monetary": r"[$£€]\s?\d+(?:[.,]\d{2,})?",
    "Emphasis": r"\b[a-zA-Z]*([a-zA-Z])\1{2,}[a-zA-Z]*\b|([!?.])\2{2,}"
}

# Extract and print 3 unique matches for each pattern
for key, pattern in regex_patterns.items():
    print(f"{key} Matches:")
    matches_found = []
    
    for text in messages:
        if key == "Emphasis":
            matches = [m.group(0) for m in re.finditer(pattern, text)]
        else:
            matches = re.findall(pattern, text)

        for match in matches:
            if key == "Emphasis" and ("www" in match.lower() or "xxx" in match.lower()): # Filter out URLs that falsely trigger the emphasis regex
                continue
                
            if match not in matches_found:
                matches_found.append(match)
            
            if len(matches_found) >= 3:
                break
                
        if len(matches_found) >= 3:
            break
            
    for match in matches_found:
        print(match)
    print()
