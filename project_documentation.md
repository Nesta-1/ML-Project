# Hackathon Project Documentation

## TABLE OF CONTENTS

| S.No. | Contents | Page no. |
| :---: | :--- | :---: |
| 1. | Abstract | |
| 2. | Introduction | |
| 3. | Literature survey | |
| 4. | Problem Statement and Objectives | |
| 5. | Methodology | |
| 6. | Results & Discussion | |
| 7. | Conclusion & Future Scope | |
| 8. | References | |

---

## 1. ABSTRACT

This project proposes a highly efficient, novel machine learning pipeline for the automated detection and classification of cyberbullying in online text, specifically targeting platforms like Twitter. While many modern solutions rely on computationally expensive deep learning or Large Language Models (LLMs), our approach leverages advanced, domain-specific feature engineering coupled with a stacking ensemble classifier to achieve competitive accuracy and interpretability. 

The core innovation lies in extracting semantic and syntactic relationships from noisy text. The pipeline employs robust text normalization to defeat common obfuscation techniques, followed by the extraction of unique features: class-aware demographic proximity (calculating the distance between demographic identifiers and negative/insult terms), interaction tokens (mapping the relationship between the target and the aggressor), and class centroid similarity. These engineered features, combined with multi-level TF-IDF representations (word, character, masked text), are fed into an optimized Stacking Classifier utilizing Support Vector Machines, Logistic Regression, Naive Bayes, and an XGBoost meta-learner. The resulting application provides rapid, accurate real-time inference suitable for scalable content moderation, resolving the tradeoff between computational efficiency and detection performance.

**Team Name:** [Your Team Name]

**Team Logo (if any):** [Insert Logo]

**Team Members:**
1. [Name, Role, Contact Information]
2. [Name, Role, Contact Information]
3. [Name, Role, Contact Information]
4. [Name, Role, Contact Information]
5. [Name, Role, Contact Information]
6. [Name, Role, Contact Information]

---

## 2. INTRODUCTION

In the era of widespread digital communication, social media platforms have increasingly become breeding grounds for toxic behavior and cyberbullying. This malicious activity—often targeting individuals based on age, gender, ethnicity, or religion—can have severe psychological impacts on victims. As the volume of user-generated content grows exponentially, manual moderation has become impossible, creating a pressing need for automated, intelligent content moderation systems.

However, detecting cyberbullying programmatically is notoriously difficult. Aggressors frequently use implicit language, sarcasm, and intentional obfuscation (e.g., replacing letters with symbols like "@sshole") to evade basic keyword filters. Furthermore, understanding the context—such as whether a negative term is directed at a specific demographic or used in self-deprecation—is crucial to avoiding false positives.

This project addresses these challenges by developing a sophisticated machine learning pipeline designed to understand the nuance of online abuse. Unlike black-box deep learning models, our approach focuses on interpretable feature engineering, capturing the exact syntactic placement of insults relative to demographic targets, and incorporating sentiment polarity. By combining these unique lexical signals with a robust ensemble modeling strategy, the system provides a highly accurate, deployable solution capable of identifying various types of cyberbullying in real-time online environments.

---

## 3. LITERATURE SURVEY

The landscape of automated cyberbullying detection has evolved significantly, transitioning from traditional keyword-based filtering to advanced machine learning and deep learning methodologies. Early research primarily utilized simple Bag-of-Words (BoW) and basic TF-IDF models paired with classifiers like Support Vector Machines (SVM) and Naive Bayes, which struggled with the nuanced and implicit nature of online abuse. Recent state-of-the-art literature has shifted heavily towards bidirectional transformer models (like RoBERTa and BERT) and complex neural architectures, which achieve high accuracy but suffer from immense computational overhead, lack of interpretability, and slow inference times. Several studies have explored the incorporation of user metadata, sentiment analysis lexicons, and dependency parsing to improve precision. Building upon this foundation, our work bridges the gap between these paradigms by bypassing the necessity of heavy deep learning models; instead, we engineer complex, context-aware semantic features—such as demographic-insult proximity and sentiment-gated interactions—and feed them into a computationally efficient, highly optimized Stacking Ensemble framework, drawing inspiration from robust classical machine learning techniques while matching modern detection standards.

---

## 4. Problem Statement and Objectives

**Problem Statement:** 
Current automated content moderation systems struggle to accurately and efficiently classify cyberbullying. Basic keyword filters are easily bypassed and lack contextual understanding, leading to numerous false positives, while advanced deep learning models are computationally prohibitive to deploy at scale, act as opaque "black boxes," and often fail to recognize targeted demographic attacks disguised by obfuscation.

**Problem Description:** 
The challenge is to build a robust, interpretable, and computationally lightweight machine learning pipeline capable of parsing noisy, real-world social media text (tweets). The system must decipher intentional text obfuscation, analyze the syntactic relationship between insults and specific demographic targets (age, gender, ethnicity, religion), and accurately categorize the type of cyberbullying without relying on slow, resource-heavy neural networks.

**Objectives:**
1. **Develop a Robust Preprocessing Engine:** Create a text normalization pipeline that effectively handles typical social media noise, including URL removal, mention masking, and reversing character obfuscation (e.g., '1' to 'i', '@' to 'a').
2. **Engineer State-of-the-Art Semantic Features:** Design and implement novel custom feature extractors, including demographic/insult proximity metrics, interaction tagging (Target vs. Self), and TF-IDF class centroid similarity scores.
3. **Train an Optimized Ensemble Model:** Build and tune a Stacking Classifier that combines the strengths of Linear SVC, Logistic Regression, and Complement Naive Bayes, utilizing an XGBoost meta-learner to achieve high accuracy and fast inference times.
4. **Deploy for Real-time Inference:** Ensure the model is optimized, cached, and integrated with a frontend user interface to allow for instantaneous cyberbullying predictions and continuous monitoring visualization.

---

## 5. METHODOLOGY

**1. Datasets Description**
The primary dataset used is a collection of categorized tweets (`cyberbullying_tweets.csv`) containing roughly 47,000 text samples heavily skewed towards identifying online harassment. The data defines classes corresponding to specific forms of bullying such as age, ethnicity, gender, and religion. For this project, general/ambiguous classes like `other_cyberbullying` were intentionally excluded to force the model to learn precise demographic-targeting signals rather than generic toxicity.

**2. Methodology: Feature Engineering & Preprocessing**
*   **Text Normalization:** A customized cleaning function standardizes texts by mapping obfuscated symbols back to basic characters, removing URLs and mentions, condensing excessive elongations, and stripping out extraneous punctuation.
*   **Structural & Interaction Masking:** The text is parsed to identify pronouns and entities, creating distinct masks for the "Target" (e.g., you, them) and the "Self" (e.g., I, me). Furthermore, an interaction token generator appends contextual tags based on the co-occurrence of these pronouns alongside predefined negative, positive, or insulting lexicons.
*   **Class-Aware Demographic Proximity:** A custom scikit-learn transformer scans the tokenized text to calculate the mathematical distance (window size) between specific demographic keywords (e.g., "muslim", "black", "woman") and toxic terms. This explicitly teaches the model *who* the insult is directed at.
*   **Class Centroid Similarity:** The pipeline calculates the TF-IDF vector of an incoming tweet and computes its cosine similarity against pre-calculated "centroid" vectors that represent the typical mathematical shape of each cyberbullying class.
*   **Numeric Density Profiling:** A statistical profile is generated for each text, calculating uppercase ratios, punctuation density, and exact VADER sentiment scores.

**3. Methodology: Model Architecture**
The system uses a Stacking Classifier pipeline. The raw text and extracted features are vectorized using a `ColumnTransformer` comprising Word TF-IDF, Character TF-IDF, and Masked text TF-IDF. This rich, multi-dimensional array is passed to three diverse base estimators: a Calibrated Linear SVC, robust Logistic Regression, and Complement Naive Bayes. Their probabilistic outputs are then fed into a highly optimized XGBoost meta-model configured for GPU acceleration.

**4. UI Integration & Caching**
To support practical application, the trained pipeline and label encoders are persistently serialized using `joblib`. Upon starting the user interface application, the serialized model is loaded into memory instantly, bypassing the training phase and allowing the UI to rapidly serve predictions and visualization metrics to end-users with near-zero latency.

---

## 6. RESULTS & DISCUSSION

The configured Stacking Ensemble model demonstrates exceptional performance, rivaling results typically achieved by much heavier Transformer-based architectures. By relying on our novel `ClassAwareLexiconFeatures` and `ClassCentroidSimilarity` transformers, the machine learning pipeline is remarkably adept at minimizing false positives—correctly recognizing the difference between a self-deprecating statement and a targeted offensive attack.

The combination of the Character TF-IDF and word-level interaction vectors provided exceptional resilience against misspelled insults and purposeful text obfuscation. While base models like the LinearSVC proved strong in parsing the high-dimensional TF-IDF data, the XGBoost meta-estimator proved critical in effectively combining the non-linear outputs of the syntactic proximity features and numeric density metrics. 

The evaluation, as demonstrated by the detailed classification reports and confusion matrices, shows highly balanced F1-scores across all difficult categories (Age, Ethnicity, Gender, Religion). Most notably, the model achieves these top-tier results while maintaining sub-second inference times and relatively low memory overhead, making it drastically more viable for edge deployment or integration into live social media APIs than prevailing deep learning alternatives.

---

## 7. CONCLUSION & FUTURE SCOPE

**Conclusion**
This project successfully designed, implemented, and validated an advanced, high-efficiency machine learning pipeline for cyberbullying detection. By shifting the focus away from black-box deep learning toward intelligent, domain-specific feature engineering (such as interaction tracking and demographic proximity metrics), the project achieved its objectives of creating a highly accurate, interpretability-friendly, and computationally economical model. The Stacking Ensemble architecture successfully synthesizes complex linguistic features to identify nuanced toxic behavior accurately and practically.

**Future Scope**
*   **Near-term:** Enhance the custom lexicons to include modern internet slang and integrate basic multilingual support to detect cross-cultural cyberbullying variants.
*   **Medium-term:** Augment the existing classical ML pipeline by concatenating lightweight pre-trained contextual embeddings (such as those from DistilRoBERTa or MiniLM) with the existing engineered features to capture complex semantic context while maintaining low computational drag.
*   **Long-term:** Package the fully trained inference pipeline into a containerized microservice API and develop a standalone browser extension, enabling real-time, proactive content masking and moderation directly on popular social media platforms.

---

## 8. REFERENCES

1. Rosa, H., Pereira, N., Ribeiro, R., et al. "Automatic cyberbullying detection: A systematic review." *Computers in Human Behavior*, 2019. [https://doi.org/10.1016/j.chb.2018.12.021](https://doi.org/10.1016/j.chb.2018.12.021)
2. Salawu, S., He, Y., & Lumsden, J. "Approaches to Automated Detection of Cyberbullying: A Survey." *IEEE Transactions on Affective Computing*, 2020. [https://ieeexplore.ieee.org/document/8141875](https://ieeexplore.ieee.org/document/8141875)
3. Al-garadi, M. A., Varathan, K. D., & Ravana, S. D. "Cybercrime detection in online communications: The experimental case of cyberbullying detection in the Twitter network." *Computers in Human Behavior*, 2016. [https://doi.org/10.1016/j.chb.2016.05.051](https://doi.org/10.1016/j.chb.2016.05.051)
4. Wulczyn, E., Thain, N., & Dixon, L. "Ex Machina: Personal Attacks Seen at Scale." *WWW '17: Proceedings of the 26th International Conference on World Wide Web*, 2017. [https://arxiv.org/abs/1610.08914](https://arxiv.org/abs/1610.08914)
5. Ptaszynski, M., Eronen, J., Masui, F., "Learning Deep on Cyberbullying is Always Better: Anomaly, Noise, and Text Representation." *Proceedings of the Workshop on Social Media Analysis*, 2019.
6. Chatzakou, D., Leontiadis, I., Blackburn, J., et al. "Mean Birds: Detecting Aggression and Bullying on Twitter." *WebSci '17*, 2017. [https://arxiv.org/abs/1702.06877](https://arxiv.org/abs/1702.06877)
7. Davidson, T., Warmsley, D., Macy, M., & Weber, I. "Automated Hate Speech Detection and the Problem of Offensive Language." *ICWSM*, 2017. [https://arxiv.org/abs/1703.04009](https://arxiv.org/abs/1703.04009)
8. Reynolds, K., Kontostathis, A., & Edwards, L. "Using Machine Learning to Detect Cyberbullying." *10th International Conference on Machine Learning and Applications (ICMLA)*, 2011.
9. Zhao, R., Zhou, A., & Mao, K. "Automatic Detection of Cyberbullying on Social Networks Based on Bullying Features." *Proceedings of the 17th International Conference on Distributed Computing*, 2016.
10. Fortuna, P., & Nunes, S. "A Survey on Automatic Detection of Hate Speech in Text." *ACM Computing Surveys*, 2018. [https://dl.acm.org/doi/10.1145/3232676](https://dl.acm.org/doi/10.1145/3232676)
11. Bozyigit, A., Utku, S., & Nasibov, E. "Cyberbullying detection: Utilizing social media features." *Expert Systems with Applications*, 2021.
12. Chen, Y., Zhou, Y., Zhu, S., & Xu, H. "Detecting Offensive Language in Social Media to Protect Adolescent Online Safety." *International Conference on Privacy, Security and Trust (PST)*, 2012.
13. Van Hee, C., Jacobs, G., Emmery, C., et al. "Automatic detection of cyberbullying in social media text." *PLoS ONE*, 2018. [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0203794](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0203794)
14. Hutto, C.J., & Gilbert, E. "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." *Eighth International AAAI Conference on Weblogs and Social Media*, 2014. 
15. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 2011.
16. Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *KDD '16*, 2016. [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
17. Nobata, C., Tetreault, J., Thomas, A., et al. "Abusive Language Detection in Online User Content." *WWW '16*, 2016.
18. MacAvaney, S., Yao, H., Yang, E., et al. "Hate speech detection: Challenges and solutions." *PLoS ONE*, 2019.
19. Zhang, Z., Robinson, D., & Tepper, J. "Detecting Hate Speech on Twitter Using a Convolution-GRU Based Deep Neural Network." *European Semantic Web Conference*, 2018.
20. Mourtaji, Y., Boukachour, M., & El Fazziki, A. "Cyberbullying detection: A systematic literature review." *IEEE International Conference on Smart Information & Communication Technologies*, 2021.
