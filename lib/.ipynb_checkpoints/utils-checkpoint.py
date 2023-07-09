import numpy as np
from scipy.special import softmax
import collections
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
import pandas as pd
import evaluate
from tqdm.auto import tqdm

def preprocess_examples(examples,tokenizer,is_test = False,max_length = 384, stride=128):
    """
    Preprocesses and tokenizes examples in preparation for inference
    
    Parameters:
    -----------
    examples : datasets.Dataset
        The dataset of examples.  Must have columns:
        'id', 'question', 'context'
    tokenizer : transformers.AutoTokenizer
        The tokenizer for the model
    is_test : bool
        True if this is the test/eval set
        False if this is the training set
    max_length : int
        The max length for context truncation
    stride : int
        The stride for context truncation

    Returns:
    --------
    inputs : dict
        The tokenized and processed data dictionary with
        keys 'input_ids', 'attention_mask', 'start_positions','end_positions'
        All values are lists of length = # of inputs output by tokenizer
            inputs['input_ids'][k] : list
                token ids corresponding to tokens in feature k
            inputs['attention_mask'][k] : list
                attention mask for feature k
            inputs['start_positions'][k] : int
                starting token positions for answer in sequence k
            inputs['end_positions'][k] : int
                ending token positions for answer in sequence k
        If is_test == True, there are additional keys 'offset_mapping' and 'example_id'
            inputs['offset_mapping'][k] : list
                modified offset mappings for feature k (entry is None if not a context token)
            inputs['example_id'][k] : int
                id of example from which feature k originated
    """
    # Some samples have lots of spaces at the end, which can
    # cause the RoBERTa tokenizer to throw truncation errors
    questions = [q.strip() for q in examples['question']]
    
    # Tokenize questions and context, making sure to only truncate context.
    # Keep truncated context with return_overflowing_tokens=True
    # Stride will help us get an uncut copy of the answer in some piece
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Context truncation produces additional features - get offset_mapping and sample_map
    answers = examples["answers"]
    if is_test:
        offset_mapping = inputs["offset_mapping"]
        example_ids = []
    else:
        offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    start_positions = []
    end_positions = []
    counter=0
    for i in range(len(inputs["input_ids"])):
        # Get the index of the sample the context piece came from
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        offset = offset_mapping[i]
        sequence_ids = inputs.sequence_ids(i)

        # If test/eval, store example_ids and modify offset_mappings
        # Will be convenient for scoring function
        if is_test:
            example_ids.append(examples["id"][sample_idx])
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        # some samples have no answer,
        # indicating the answer is not in any
        # context piece, and so all corresponding
        # context pieces should get label (0,0)
        if len(answer['answer_start'])==0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        # If there is an answer, record its start and end
        # character index and get the sequence_ids
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        
        # Find the start and end of the context piece
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context piece, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise find the starting and ending positions of the answer
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    if is_test:
        inputs["example_id"] = example_ids
    return inputs

def make_predictions(model,tokenizer,inputs,batch_size=16):
    """
    Generates lists of logits for possible starting
    and ending positions of answer

    Parameters:
    -----------
    model : transformers.AutoModelForQuestionAnswering
        The trained model
    tokenizer : transformers.AutoTokenizer
        The model's tokenizer
    inputs : dataset
        The tokenized and and preprocessed dataset containing columns
        'input_ids' and 'attention_mask' (additional columns are ignored)

    Returns:
    --------
    start_logits, end_logits : np.array
        Arrays 
    """  
    # Set mps or cuda device if available
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # The model only expects 'input_ids' and 'attention_mask'
    remove_cols = [col for col in inputs.column_names\
                   if col not in ['input_ids','attention_mask']]
    data_for_model = inputs.remove_columns(remove_cols)
    data_for_model.set_format("torch",device=device)
    
    # Initialize dataloader
    dl = DataLoader(
        data_for_model,
        collate_fn=default_data_collator,
        batch_size=batch_size,
    )
    start_logits, end_logits = [],[]
    # Make predictions and export logits as np.array
    model = model.to(device)
    for batch in dl:
        outputs = model(**batch)
        start_logits.append(outputs.start_logits.cpu().detach().numpy())
        end_logits.append(outputs.end_logits.cpu().detach().numpy())
    
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    
    return start_logits, end_logits

def format_predictions(start_logits, end_logits, inputs, examples,
                      n_best = 20,max_answer_length=30,
                      convert_empty = False):
    """
    Postprocessing of logits into prediction data
    Parameters:
    -----------
    start_logits, end_logits : list, list
        sequences of logits corresponding to possible start
        and end token indices of the answer
    inputs : dataset
        The tokenized and and preprocessed dataset containing columns
        'example_id', 'offset_mapping' (other columns are ignored)
    examples : datasets.Dataset
        The dataset of examples.  Must have columns:
        'id', 'question', 'context'
    n_best : int
        The number of top start/end (by logit) indices to consider
    max_answer_length : int
        The maximum length (in characters) allowed for a candidate answer
    convert_empty : bool
        Whether to transform prediction text of "" to
        "I don't have an answer based on the context provided."
        
    Returns:
    --------
    predicted_answers : list(dict)
        for each entry, keys are 'id', 'prediction_text'
    """
    assert n_best <= len(inputs['offset_mapping'][0]), 'n_best cannot be larger than max_length'
    
    # Dictionary whose keys are example ids and values are
    # corresponding indices of tokenized feature sequences
    example_to_inputs = collections.defaultdict(list)
    for idx, feature in enumerate(inputs):
        example_to_inputs[feature["example_id"]].append(idx)
    
    # Loop through examples
    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []
        
        # For each example, loop through corresponding features
        for feature_index in example_to_inputs[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            
            # Retrieve modified offset_mapping;
            # Context tokens indices have actual offset_mapping pair,
            # all other indices have None
            offsets = inputs[feature_index]['offset_mapping']
            
            # Get indices of n_best most likely start, end token
            # index values for the answer
            start_indices = np.argsort(start_logit)[-1:-n_best-1:-1].tolist()
            end_indices = np.argsort(end_logit)[-1 :-n_best-1: -1].tolist()

            # Loop over all possible start,end pairs
            for start_index in start_indices:
                for end_index in end_indices:
                    # Skip pair which would require an answer to have
                    # negative length or length greater than max_answer_length
                    if(
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    
                    # Skip pairs having None for exactly one of offsets[start_index],
                    # offsets[end_index] - which would require the answer to only
                    # partially lie in this context sequence 
                    if (offsets[start_index] is None)^(offsets[end_index] is None):
                        continue
                    
                    # Pairs which have None for both correspond to an
                    # empty string as the answer prediction
                    # Adding logits is equivalent to multiplying probabilities
                    if (offsets[start_index] is None)&(offsets[end_index] is None):
                        answers.append(
                            {
                                    "text": '',
                                    "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )
                    # If neither are None and the answer has positive length less than
                    # max_answer_length, then this corresponds to a non-empty answer candidate
                    # in the context and we want to include it in our list
                    else:
                        answers.append(
                            {
                                "text": context[offsets[start_index][0] : offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )
        # Retrieve logits and probability scores for all viable start,end combinations
            
        # If there are candidate answers, choose the candidate with largest logit score
        # this might be ''
        if len(answers)>0:
            best_answer = max(answers, key=lambda x:x['logit_score'])
            predicted_answers.append({'id':example_id, 'prediction_text':best_answer['text']})
        else:
            predicted_answers.append({'id':example_id, 'prediction_text':''})
        if convert_empty:
            for pred in predicted_answers:
                if pred['prediction_text'] == '':
                    pred['prediction_text'] = "I don't have an answer based on the context provided."
    return predicted_answers

def compute_metrics(start_logits, end_logits, inputs, examples,
                    n_best = 20, max_answer_length=30):
    """
    Compute the results of the SQuAD v2 metric on predictions
    Parameters:
    -----------
    start_logits, end_logits : list, list
        sequences of logits corresponding to possible start
        and end token indices of the answer
    inputs : dataset
        The tokenized and and preprocessed dataset containing columns
        'example_id', 'offset_mapping' (other columns are ignored)
    examples : datasets.Dataset
        The dataset of examples.  Must have columns:
        'id', 'question', 'context'
    n_best : int
        The number of top start/end (by logit) indices to consider
    max_answer_length : int
        The maximum length (in characters) allowed for a candidate answer
    Returns:
    --------
    metrics : dict
        dictionary of metric values
    """
    metric = evaluate.load('squad_v2')
    predicted_answers = format_predictions(start_logits, end_logits, inputs, examples,
                                           n_best = n_best,max_answer_length=max_answer_length)
    for pred in predicted_answers:
        pred['no_answer_probability'] = 1.0 if pred['prediction_text']=='' else 0.0
    theoretical_answers = [{'id':ex['id'],'answers':ex['answers']} for ex in examples]
    return metric.compute(predictions = predicted_answers, references = theoretical_answers)