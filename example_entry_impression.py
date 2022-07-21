"""
Each entry in personachat is a dict with two keys personality and utterances, the dataset is a list of entries(dialogs).
personality:  list of strings containing the personality of the agent
utterances: lists of strings containing the utterances in a dialog corresponding to the personality.
Preprocessing:
    - Spaces before periods at end of sentences
    - everything lowercase
"""

EXAMPLE_ENTRY = {
    "personality": ["i like to remodel homes .", "i like to go hunting .",
                    "i like to shoot a bow .", "my favorite holiday is halloween ."],
    "utterances": ["you must be very fast . hunting is one of my favorite hobbies .",
                   "i also remodel homes when i am not out bow hunting .",
                   "that is awesome . do you have a favorite season or time of year ?",
                   "what is your favorite meat to eat ?",
                   "i like chicken or macaroni and cheese .",
                   "i am going to watch football . what are you canning ?",
                   "if i have time outside of hunting and remodeling homes . which is not much !"]
}