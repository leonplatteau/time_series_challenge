#This is the second attempt with the aim of building a more advanced classification model, based on a 2D-CNN 
# In this attempt, multiple aspects are implemented, trying to make a complete as possible model with the lowest KL-div
# Build-up of the code:
# (0) Initialization: loading all packages, loading all files, defining invariant parameters.
# (1) Preprocessing1: defining extra necessary columns in dataframe:
#       This is mainly on labels and amount of votes
# (2) Preprocessing2: filtering the EEG signals (bandpass 0.5-40Hz --> in literature (https://www.researchgate.net/publication/384165067_Advancing_Emotional_Health_Assessments_A_Hybrid_Deep_Learning_Approach_Using_Physiological_Signals_for_Robust_Emotion_Recognition))
# (3) Representation
# (4) Appplying the actual CNN-model
# (5) Training: loop with layer1 and layer2, learning-rate, checkpoints, mixed precision
# (6) Cross-validation.
# (7) Inference and submission
# This is possibly the final attempt, therefore I try to make it as complete as possible. 

