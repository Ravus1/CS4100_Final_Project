import nltk
import glob
import os
from datasets import load_dataset
from rouge_score import rouge_scorer

# Download 'punkt' if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

MAX_SENTENCES_PER_DOC = 120

def load_and_process_cnn_dailymail(sample_size=400):
    """
    Downloads and processes the CNN/Daily Mail dataset.
    It programmatically labels sentences based on their ROUGE score
    with the summary.
    """
    print("Loading CNN/Daily Mail dataset...")
    # Load a subset of the dataset for demonstration purposes
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{sample_size}]")
    
    all_sentences = []
    all_labels = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    print(f"Processing {sample_size} articles...")
    for i, example in enumerate(dataset):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{sample_size} articles")
        
        article_text = example['article']
        summary_text = example['highlights']
        
        sentences = nltk.sent_tokenize(article_text)
        # Optionally cap very long articles so more documents contribute
        if len(sentences) > MAX_SENTENCES_PER_DOC:
            sentences = sentences[:MAX_SENTENCES_PER_DOC]
        if not sentences:
            continue

        # Calculate ROUGE scores for each sentence against the summary
        scores = [scorer.score(summary_text, sent)['rougeL'].fmeasure for sent in sentences]
        
        # Label more sentences per document as important (increase top-k)
        k = min(5, len(sentences))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        labels = [1 if i in top_indices else 0 for i in range(len(sentences))]

        all_sentences.extend(sentences)
        all_labels.extend(labels)

    print("Finished processing CNN/Daily Mail dataset.")
    return all_sentences, all_labels

def load_and_process_hf_summarization_dataset(
    dataset_name,
    config_name=None,
    text_field="document",
    summary_field="summary",
    split="train[:500]"
):
    """
    Generic loader for Hugging Face text+summary datasets.

    - dataset_name: HF datasets ID, e.g. "booksum" or a processed lecture dataset.
    - config_name: optional HF config name.
    - text_field: field in the dataset containing the full text / lecture.
    - summary_field: field containing the gold summary / notes.
    - split: HF split string, e.g. "train[:100]" for a small subset.
    """
    print(f"Loading HF dataset '{dataset_name}' (config={config_name}, split={split})...")
    if config_name is None:
        ds = load_dataset(dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name, config_name, split=split)

    all_sentences = []
    all_labels = []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for i, ex in enumerate(ds):
        text = ex[text_field]
        summary = ex[summary_field]

        sentences = nltk.sent_tokenize(text)
        # Optionally cap very long documents so more documents contribute
        if len(sentences) > MAX_SENTENCES_PER_DOC:
            sentences = sentences[:MAX_SENTENCES_PER_DOC]
        if not sentences:
            continue

        scores = [
            scorer.score(summary, sent)["rougeL"].fmeasure
            for sent in sentences
        ]

        # Label the top‑k sentences as important (increase k for more positives)
        k = min(5, len(sentences))
        top_indices = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:k]
        labels = [1 if j in top_indices else 0 for j in range(len(sentences))]

        all_sentences.extend(sentences)
        all_labels.extend(labels)

    print(f"Finished processing HF dataset '{dataset_name}'.")
    return all_sentences, all_labels

def load_local_lecture_notes():
    """
    Loads all local lecture notes, tokenizes them, and provides manually labeled training data.
    """
    data_path = "data/"
    all_sentences = []
    all_labels = []

    # A dictionary mapping filenames to their important sentences for labeling.
    important_sentences_map = {
        "lecture_notes.txt": [
            "However, many real-life problems do not focus on the specific path taken to reach the goal.",
            "With this in mind, we now turn our attention to local search methods.",
            "Local search algorithms offer two key advantages in scenarios where the path to the goal is inconsequential:",
            "Hill-climbing is a type of local search algorithm.",
            "Hill climbing iteratively moves in either an increasing/ascent or decreasing/descent direction based on the problem objective.",
            "This particular approach is one of several related variants of hill-climbing, and is called Steepest-Ascent Hill Climbing.",
            "This latter approach is known as First-Choice Hill Climbing.",
            "This final variant incorporates randomness in a slightly different way; it chooses randomly among all uphill/downhill moves, with the probability of selecting a particular move varying with the steepness of the ascent/descent.",
            "Hill-climbing approaches, while often effective, may fail to produce an optimal solution in certain instances.",
            "This may happen as a result of the search getting stuck in a local optimum.",
            "One of the most popular methods to circumvent getting stuck in local optima is to randomly restart the hill-climbing process with a different initial state.",
            "A second way to wiggle our way out of local optima is to allow steps that lead to a worse objective function value with a small probability."
        ],
        "lecture_notes_ml_intro.txt": [
            "A series of algorithms classified as machine learning models, our goal is to enable computers to learn from and make predictions or decisions based on data.",
            "Machine learning models are often classified into supervised learning and unsupervised learning.",
            "In supervised learning, models are trained on labeled data, where the input and output are known.",
            "In unsupervised learning on the other hand, models are trained on unlabeled data, and the goal is to discover patterns or relationships in the data.",
            "If most of AI could be summarized as a single mathematical process, it would be gradient descent.",
            "The gradient is a vector that points in the direction of the greatest rate of increase of the objective function.",
            "This is known as gradient descent.",
            "The chain rule is a fundamental concept in calculus that allows us to compute the derivative of a composite function.",
            "In order to efficiently compute derivatives of complicated objective functions, we rely on an internal representation of the objective function in the form of a computational graph.",
            "This process is known as automatic differentiation, and is central to modern machine learning frameworks.",
            "This process involves selecting key characteristics from the data that are most relevant for the task at hand, often based on a domain knowledge, and converting them into numerical representations, which we can then use in order to compute complex relationships.",
            "This process is called feature extraction.",
            "The process of extracting features and converting them into a numerical vector is called vectorization."
        ],
        "lecture_notes_classifiers.txt": [
            "Naive Bayes classification is a simple probabilistic classifier based on the application of Bayes' theorem, where we (naively) assume that features are independent.",
            "To account for this, we can use Laplace smoothing, where we add a small constant to each count to ensure that no probability is ever 0.",
            "To tackle this problem, we now build a more general framework for classification, using linear decision boundaries.",
            "A linear classifier or decision boundary in two dimensions is a line that separates the class -1 datapoints from the class 1 datapoints.",
            "In machine learning, we call this quantity the margin - which is defined as the distance between the decision boundary and the closest point of the two classes in case of binary classification.",
            "The process of finding the decision boundary that maximizes the margin is called maximal margin classification, and is a key idea in the field of machine learning.",
            "The decision boundary that maximizes the margin is called the maximum margin hyperplane, and the algorithm that finds this boundary is called the support vector machine (SVM).",
            "The fundamental idea is to find a linear separator that not only classifies our data correctly, but also maximizes the margin between the two classes.",
            "These points are called the support vectors (hence the name!).",
            "This can be captured using the hinge loss function, which is defined as max(0, 1 - y_i(w^T x_i + b)).",
            "Despite its name, logistic regression is a classification algorithm, and is used to predict the probability of a data point belonging to a particular class.",
            "This function is called the sigmoid function, and is defined as σ(z) = 1 / (1 + e^(-z)).",
            "This is referred to as the Binary Cross Entropy (BCE) loss."
        ],
        "lecture_notes_neural_networks.txt": [
            "Neural networks are a class of machine learning models that are inspired by the structure and function of the human brain.",
            "A perceptron is a simple mathematical model, which takes a set of inputs, performs a weighted sum of these inputs, adds a bias term, and passes the result through an activation function to produce an output.",
            "An activation function is a non-linear function that introduces non-linearity into the model, and is what allows neural networks to learn very complex patterns in the data.",
            "The figure above shows the XOR problem, which is a classic example of a problem that cannot be solved by a single linear classifier.",
            "Neural networks, by combining multiple perceptrons in hidden layers, are able to learn such non-linear decision boundaries.",
            "The process of training a model involves finding the weights and biases that minimize a loss function that captures how badly our model is performing based on our current estimates of its parameters.",
            "This is typically done using gradient descent.",
            "The loss function that we wish to minimize is, thus, once again the Binary Cross Entropy (BCE) loss, which we've already seen in Logistic Regression.",
            "However, in many real-world scenarios, we wish to classify data into more than two classes.",
            "This is called multiclass classification.",
            "To convert the outputs to a valid probability distribution, the output of the network is passed through a softmax function."
        ],
        "lecture_notes_probabilistic_reasoning.txt": [
            "To address this, we leverage the Markov assumption, which states that the probability of the current state depends only on a finite series of previous states.",
            "As a special case, this assumption gives rise to the Markov chain representation, asserting that future state predictions are conditionally independent of the past given the present state- P(X_t | X_{t-1}, ..., X_0) = P(X_t | X_{t-1}) .",
            "When t -> infinity, we notice that our probabilities converge to what is known as the stationary distribution, following which the distribution does not change - it satisfies the equation p' = p'P.",
            "Instead, our agents gather observations over time, which they use to update their beliefs about the environment.",
            "A Hidden Markov Model consists of three fundamental components: the initial distribution p(X_0), transition probabilities denoted by P(X_t | X_{t-1}), and observation/emission probabilities P(E_t | X_t).",
            "Temporal Dependency: The future states of the system depend solely on the present state, encapsulated by the transition probabilities P(X_t | X_{t-1}).",
            "Observational Independence: Given the current state, the observation at a particular time step is independent of all other observations and states in the sequence, apart from the current state.",
            "To this end, we use the Forward Algorithm which has a time complexity of O(S^2 * T) which uses a lookup table to store intermediate values as it builds up the probability of the observation sequence.",
            "For the second type of reasoning, i.e., finding the most likely hidden-state sequence, given an observation sequence, we use a dynamic programming algorithm, called the Viterbi algorithm."
        ],
        "lecture_notes_mdps.txt": [
            "This stochastic system can be formally represented as a Markov Decision Process (MDP), which is a mathematical model describing a problem characterized by uncertainty.",
            "Finally, a policy pi is defined as a mapping from states to actions.",
            "The goal of the agent in an MDP is to find the policy that maximizes the expected cumulative reward over time.",
            "This is known as the optimal policy.",
            "This is called the value of the state under the policy, and denoted V^pi (s).",
            "This is given by the following Bellman equation:",
            "This algorithm is called the Policy Evaluation algorithm, and works as follows.",
            "This is known as the policy iteration algorithm.",
            "Such scenarios can be modeled using the Partially Observable MDP (or POMDP) framework.",
            "The belief state is defined as a probability distribution over the state space, representing the agent's likelihood of being in each state in the state-space."
        ],
        "lecture_notes_rl.txt": [
            "In such environments, we deploy a class of algorithms known collectively as reinforcement learning (RL), where the agent learns to make decisions through interactions with the environment.",
            "Such an approach is called Model Based Monte Carlo (MBMC) estimation, since we assume an underlying MDP, and use the data solely to infer the model's parameters, namely the transition probabilities and rewards.",
            "The solution - Model-Free Monte Carlo (MFMC) methods.",
            "As the name suggests, in MFMC, we use the data from the environment to directly estimate Q-values, without first constructing the underlying MDP.",
            "This, in turn, leads us to a new class of RL algorithms, collectively known as Temporal Difference methods.",
            "Finally, to recycle a metaphor from the 17th century, combining these two ideas (bootstrapping and convex combinations) finally gives us the SARSA algorithm, the stone that kills two birds.",
            "SARSA, therefore, is an example of an on-policy learning technique.",
            "To this end, we now dive into off-policy learning, where the agent learns the expected value for the optimal policy while using a completely different policy to explore the environment.",
            "This gives us the following algorithm to infer optimal expected utilities:",
            "One popular technique ussed to account for these opposing objectives (exploration v/s exploitation) is the e-greedy approach."
        ],
        "lecture_notes_deep_learning_vision.txt": [
            "A Convolutional Neural Network (CNN) is a class of deep neural networks, most commonly applied to analyzing visual imagery.",
            "The core building block of a CNN.",
            "This layer is used to reduce the spatial dimensions (width and height) of the input volume for the next convolutional layer.",
            "After several convolutional and pooling layers, the high-level features are flattened into a one-dimensional vector and fed into one or more fully connected layers.",
            "Image classification is a fundamental task in computer vision where the goal is to assign a label to an input image from a predefined set of categories.",
            "Object detection is a more challenging task than image classification.",
            "It involves not only identifying which objects are in an image but also locating them by drawing a bounding box around each object."
        ],
        "lecture_notes_deep_learning_nlp.txt": [
            "Recurrent Neural Networks (RNNs) are a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.",
            "Long Short-Term Memory (LSTM) networks are a special kind of RNN, designed to solve the vanishing gradient problem.",
            "The Transformer architecture, introduced in the paper \"Attention Is All You Need,\" proposed a new approach that relies entirely on the attention mechanism, dispensing with recurrence altogether.",
            "The attention mechanism allows the model to weigh the importance of different words in the input sequence when producing an output.",
            "The Transformer has become the state-of-the-art for many NLP tasks and is the foundation for popular pre-trained models like BERT and GPT."
        ],
        "lecture_notes_unsupervised_learning.txt": [
            "Unsupervised learning is a type of machine learning where models are trained on unlabeled data, and the goal is to discover patterns or relationships in the data.",
            "Clustering is a common unsupervised learning task that involves grouping a set of data points in such a way that points in the same group (called a cluster) are more similar to each other than to those in other clusters.",
            "K-Means is one of the most popular clustering algorithms.",
            "Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters.",
            "Dimensionality reduction is the process of reducing the number of random variables under consideration by obtaining a set of principal variables.",
            "PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.",
            "t-SNE is a machine learning algorithm for visualization."
        ],
        "lecture_notes_model_evaluation.txt": [
            "In machine learning, model evaluation is the process of using different evaluation metrics to understand a machine learning model's performance, as well as its strengths and weaknesses.",
            "For classification tasks, several metrics are used to evaluate a model's performance.",
            "Cross-validation is a resampling technique used to evaluate machine learning models on a limited data sample.",
            "Regularization is a set of techniques used to prevent overfitting in machine learning models.",
            "L1 regularization adds a penalty equal to the absolute value of the magnitude of the coefficients.",
            "L2 regularization adds a penalty equal to the square of the magnitude of the coefficients."
        ],
        "lecture_notes_ai_ethics.txt": [
            "AI ethics is a branch of ethics that studies the moral issues and implications of artificial intelligence.",
            "Fairness in AI refers to the goal of preventing AI systems from making decisions that are biased or discriminatory against certain individuals or groups.",
            "Accountability in AI means ensuring that there are clear lines of responsibility for the outcomes of AI systems.",
            "Transparency is closely related to accountability and refers to the need for AI systems to be understandable and explainable.",
            "Explainable AI (XAI) is a field of research that focuses on developing techniques to make AI models more transparent.",
            "AI systems often require large amounts of data to be trained, which raises significant privacy concerns."
        ]
    }

    lecture_files = glob.glob(os.path.join(data_path, "lecture_notes*.txt"))

    for file_path in lecture_files:
        notes = load_lecture_notes(file_path)
        if not notes:
            continue

        sentences = nltk.sent_tokenize(notes)
        file_name = os.path.basename(file_path)
        
        important_sentences = important_sentences_map.get(file_name, [])
        
        labels = [1 if s in important_sentences else 0 for s in sentences]

        all_sentences.extend(sentences)
        all_labels.extend(labels)

    return all_sentences, all_labels

def get_training_data(
    use_cnn_dailymail=True,
    use_hf_lecture_dataset=False,
    hf_dataset_name=None,
    hf_config_name=None,
    hf_text_field="document",
    hf_summary_field="summary",
    hf_split="train[:500]",
    sample_size=400,
):
    """
    Main function to get training data.

    - use_cnn_dailymail: include CNN/Daily Mail data.
    - use_hf_lecture_dataset: include an extra HF summarization dataset (e.g. lecture/talk style).
    - hf_*: options for the HF dataset.
    - If both are False or no data is loaded, falls back to local lecture notes.
    """
    sentences = []
    labels = []

    if use_cnn_dailymail:
        s_cnn, y_cnn = load_and_process_cnn_dailymail(sample_size=sample_size)
        sentences.extend(s_cnn)
        labels.extend(y_cnn)

    if use_hf_lecture_dataset and hf_dataset_name is not None:
        s_hf, y_hf = load_and_process_hf_summarization_dataset(
            dataset_name=hf_dataset_name,
            config_name=hf_config_name,
            text_field=hf_text_field,
            summary_field=hf_summary_field,
            split=hf_split,
        )
        sentences.extend(s_hf)
        labels.extend(y_hf)

    if not sentences:
        return load_local_lecture_notes()

    # Print basic label distribution for monitoring
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    print(f"Label distribution: {num_pos} positives, {num_neg} negatives "
          f"({num_pos / max(1, len(labels)):.3f} positive fraction)")

    return sentences, labels

