# utils/text_utils.py

BULLYING_WORDS = [
    # Common insults and slurs
    "idiot", "stupid", "dumb", "loser", "moron", "retard", "fool",
    "hate", "ugly", "trash", "worthless", "lame", "jerk",
    
    # Swear words and curse words (mild to strong)
    "damn", "shit", "crap", "bitch", "asshole", "fuck", "fucking", "bastard",
    "dick", "piss", "cock", "bollocks", "bugger", "arse", "prick",
    
    # Threatening or aggressive words
    "kill", "die", "destroy", "suck", "shut up", "shut the fuck up", "get lost",
    
    # Racial/ethnic slurs (use with care)
    "chink", "nigger", "kike", "spic", "gook", "raghead",
    
    # Sexual harassment related
    "slut", "whore", "cunt", "bitch", "twat",
    
    # Common toxic phrases or derogatory terms
    "loser", "idiotic", "pathetic", "scum", "trash", "worthless",
    
    # Emojis often used aggressively (just words for matching)
    ":angry_face:", ":middle_finger:", ":rage:", ":skull:", ":poop:"
]

def extract_toxic_words(text):
    """Extract toxic words found in input text from the predefined list."""
    text_lower = text.lower()
    found = [word for word in BULLYING_WORDS if word in text_lower]
    return list(set(found))  # unique matches
