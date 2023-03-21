import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle
from os import path as check_path
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score as prec_score
from sklearn.metrics import recall_score as recall_score
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertConfig, BertModel, BertPreTrainedModel, get_linear_schedule_with_warmup, AdamW, BertTokenizerFast
from torch.nn import LayerNorm as BertLayerNorm


from ..utils.data_editor import get_data, get_classes, get_predictions
from ..utils.utils_cage import probability, log_likelihood_loss, precision_loss, predict_gm_labels
from ..utils.utils_jl import log_likelihood_loss_supervised, entropy, kl_divergence
from .models.models import *
from transformers import LayoutLMTokenizer,LayoutLMForTokenClassification
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
import pickle 
from torch.nn import CrossEntropyLoss
max_seq_length = 512
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#[Note: 
	#1. Loss function number, Calculated over, Loss function:
	#		1, L, Cross Entropy(prob_from_feature_model, true_labels)
	#		2, U, Entropy(prob_from_feature_model)
	#		3, U, Cross Entropy(prob_from_feature_model, prob_from_graphical_model)
	#		4, L, Negative Log Likelihood
	#		5, U, Negative Log Likelihood(marginalised over true labels)
	#		6, L and U, KL Divergence(prob_feature_model, prob_graphical_model)
	#		7, Quality guide
	#
	#2. each pickle file should follow the standard convention for data storage]
	#
	#3. shapes of x,y,l,s:
	#	x: [num_instances, num_features], feature matrix
	#	y: [num_instances, 1], true labels, if available
	#	l: [num_instances, num_rules], 1 if LF is triggered, 0 else
	#	s: [num_instances, num_rules], continuous score
#]
class CordDataset(Dataset):
    def __init__(self, examples, tokenizer, labels, pad_token_label_id,n_lfs):
        features = convert_examples_to_featuresz(
            examples,
            labels,
            max_seq_length,
            tokenizer,
	    	n_lfs,
            cls_token_at_end=False,
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            pad_token_label_id=pad_token_label_id,
            
        )

        self.features = features
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )
        self.all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)
        # print(features)		
        # z=[f.L for f in features]
        # z=torch.tensor(z,dtype=torch.long)
        # print(z)
        self.all_L = torch.tensor([f.L for f in features], dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index],
			self.all_L[index],
        )

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        boxes,
        L
    ):
        assert (
            0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes
        self.L = L


def convert_examples_to_featuresz(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    n_lfs,    
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
    pad_token_segment_id=0,
    pad_token_label_id=-1,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
	# ABSTAIN=[None,None,None,None,None,None,None]
	# ABSTAIN=[3,3,3,3,3,3,3]
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    ABSTAIN=[-1]*n_lfs
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for i in range(len(examples[0])):
        width, height = 1000, 1000
        words = examples[0]
        labels = examples[1]
        boxes = examples[2]
        L=examples[3]
        tokens = []
        token_boxes = []
        label_ids = []
        token_L=[]
        for word, label, box,L in zip(
            words[i], labels[i], boxes[i],L[i]
        ):
            if len(word) < 1: # SKIP EMPTY WORD
              continue
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend(
                [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            token_L.extend([L] * len(word_tokens))	

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            token_L = token_L[: (max_seq_length - special_tokens_count)]
        tokens += [sep_token]
        token_boxes += [sep_token_box]
        label_ids += [pad_token_label_id]
        token_L += [ABSTAIN]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            token_boxes += [sep_token_box]
            label_ids += [pad_token_label_id]
            token_L += [ABSTAIN]
        segment_ids = [sequence_a_segment_id] * len(tokens) 

        if cls_token_at_end:
            tokens += [cls_token]
            token_boxes += [cls_token_box]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            token_L += [ABSTAIN]
			
        else:
            tokens = [cls_token] + tokens
            token_boxes = [cls_token_box] + token_boxes
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            token_L = [ABSTAIN] + token_L

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            token_boxes = ([pad_token_box] * padding_length) + token_boxes
            token_L += ([ABSTAIN] *  padding_length) + token_L
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            token_boxes += [pad_token_box] * padding_length
            token_L += [ABSTAIN] * padding_length
			

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length
        # assert len(L) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                boxes=token_boxes,
				L=token_L,
            )
        )
    return features

class JL:
	'''
	Joint_Learning class:
		[Note: from here on, feature model(fm) and feature-based classification model are used interchangeably. graphical model(gm) and CAGE algorithm terms are used interchangeably]
		Loss function number | Calculated over | Loss function: (useful for loss_func_mask in fit_and_predict_proba and fit_and_predict functions)
			1, L, Cross Entropy(prob_from_feature_model, true_labels)
			2, U, Entropy(prob_from_feature_model)
			3, U, Cross Entropy(prob_from_feature_model, prob_from_graphical_model)
			4, L, Negative Log Likelihood
			5, U, Negative Log Likelihood(marginalised over true labels)
			6, L and U, KL Divergence(prob_feature_model, prob_graphical_model)
			7, _,  Quality guide
	
	Args:
		path_json: Path to json file containing the dictionary of number to string(class name) map
		n_lfs: number of labelling functions used to generate pickle files
		n_features: number of features for each instance in the first array of pickle file aka feature matrix
		feature_model: The model intended to be used for features, allowed values are 'lr'(Logistic Regression) or 'nn'(Neural network with 2 hidden layer) string, default is 'nn'
		n_hidden: Number of hidden layer nodes if feature model is 'nn', type is integer, default is 512
	'''
	def __init__(self, path_json, n_lfs, n_features, feature_model = 'layoutlm', n_hidden = 512):
		assert type(path_json) == str
		assert type(n_lfs) == np.int or type(n_lfs) == np.float
		assert type(n_features) == np.int or type(n_features) == np.float
		assert type(n_hidden) == np.int or type(n_hidden) == np.float
		assert feature_model == 'layoutlm'
		
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda:1" if use_cuda else "cpu")
		torch.backends.cudnn.benchmark = True
		torch.set_default_dtype(torch.float64)

		self.class_dict = get_classes(path_json)
		self.class_list = list((self.class_dict).keys())
		self.class_list.sort()
		self.n_classes = len(self.class_dict)

		self.class_map = {value: index for index, value in enumerate(self.class_list)}
		self.class_map[None] = -1
		print(self.class_map)

		self.n_lfs = int(n_lfs)
		self.n_hidden = int(n_hidden)
		self.feature_based_model = feature_model
		self.n_features = n_features
		self.k, self.continuous_mask = None, None

		self.pi = torch.ones((self.n_classes, self.n_lfs), device = self.device).double()
		(self.pi).requires_grad = True
		self.theta = torch.ones((self.n_classes,self.n_lfs), device = self.device).double()
		(self.theta).requires_grad = True

		if self.feature_based_model == 'layoutlm':
			self.feature_model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=self.n_classes)
			self.feature_model.to(device = self.device)
		else:
			print('Error: JL class - unrecognised feature_model in initialisation')
			exit(1)

		self.fm_optimal_params = deepcopy((self.feature_model).state_dict())
		self.pi_optimal, self.theta_optimal = (self.pi).detach().clone(), (self.theta).detach().clone()

	def save_params(self, save_path):
		'''
			member function to save parameters of JL
		Args:
			save_path: path to pickle file to save parameters
		'''
		file_ = open(save_path, 'wb')
		pickle.dump(self.theta, file_)
		pickle.dump(self.pi, file_)
		pickle.dump((self.feature_model).state_dict(), file_)
		pickle.dump(self.theta_optimal, file_)
		pickle.dump(self.pi_optimal, file_)
		pickle.dump((self.fm_optimal_params), file_)
		pickle.dump(self.n_classes, file_)
		pickle.dump(self.n_lfs, file_)
		pickle.dump(self.n_features, file_)
		pickle.dump(self.n_hidden, file_)
		pickle.dump(self.feature_based_model, file_)
		file_.close()
		return

	def load_params(self, load_path):
		'''
			member function to load parameters to JL
		Args:
			load_path: path to pickle file to load parameters
		'''
		assert check_path.exists(load_path)
		file_ = open(load_path, 'rb')
		self.theta = pickle.load(file_)
		self.pi = pickle.load(file_)
		fm_params = pickle.load(file_)
		(self.feature_model).load_state_dict(fm_params)

		self.theta_optimal = pickle.load(file_)
		self.pi_optimal = pickle.load(file_)
		self.fm_optimal_params = pickle.load(file_)

		assert self.n_classes == pickle.load(file_)
		assert self.n_lfs == pickle.load(file_)
		assert self.n_features == pickle.load(file_)
		temp_n_hidden = pickle.load(file_)
		temp_feature_based_model = pickle.load(file_)
		assert self.feature_based_model == temp_feature_based_model
		if temp_feature_based_model == 'layoutlm':
			assert self.n_hidden == temp_n_hidden
		
		file_.close()

		assert (self.pi).shape == (self.n_classes, self.n_lfs)
		assert (self.theta).shape == (self.n_classes, self.n_lfs)
		assert (self.pi_optimal).shape == (self.n_classes, self.n_lfs)
		assert (self.theta_optimal).shape == (self.n_classes, self.n_lfs)

		return

		

	def fit_and_predict_proba(self, path_L, path_U, path_V, path_T,train,train_u,dev,test, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, path_log = None, return_gm = False, n_epochs = 100, start_len = 7,\
	 stop_len = 10, is_qt = True, is_qc = True, qt = 0.9, qc = 0.85, metric_avg = 'binary'):
		'''
		Args:
			path_L: Path to pickle file of labelled instances
			path_U: Path to pickle file of unlabelled instances
			path_V: Path to pickle file of validation instances
			path_T: Path to pickle file of test instances
			loss_func_mask: list of size 7 where loss_func_mask[i] should be 1 if Loss function (i+1) should be included, 0 else. Checkout Eq(3) in :cite:p:`DBLP:journals/corr/abs-2008-09887`
			batch_size: Batch size, type should be integer
			lr_fm: Learning rate for feature model, type is integer or float
			lr_gm: Learning rate for graphical model(cage algorithm), type is integer or float
			use_accuracy_score: The score to use for termination condition on validation set. True for accuracy_score, False for f1_score
			path_log: Path to log file to append log. Default is None which prints accuracies/f1_scores is printed to terminal
			return_gm: Return the predictions of graphical model? the allowed values are True, False. Default value is False
			n_epochs: Number of epochs in each run, type is integer, default is 100
			start_len: A parameter used in validation, refers to the least epoch after which validation checks need to be performed, type is integer, default is 7
			stop_len: A parameter used in validation, refers to the least number of continuous epochs of non incresing validation accuracy after which the training should be stopped, type is integer, default is 10
			is_qt: True if quality guide is available(and will be provided in 'qt' argument). False if quality guide is intended to be found from validation instances. Default is True
			is_qc: True if quality index is available(and will be provided in 'qc' argument). False if quality index is intended to be found from validation instances. Default is True
			qt: Quality guide of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.9
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			metric_avg: Average metric to be used in calculating f1_score/precision/recall, default is 'binary'
		Return:
			If return_gm is True; the return value is two predicted labels of numpy array of shape (num_instances, num_classes), first one is through feature model, other one through graphical model.
			Else; the return value is predicted labels of numpy array of shape (num_instances, num_classes) through feature model. For a given model i,j-th element is the probability of ith instance being the 
		  	jth class(the jth value when sorted in ascending order of values in Enum) using that model. It is suggested to use the probailities of feature model
		'''


		train = pickle.load(open(train, 'rb'))
		train_u = pickle.load(open(train_u, 'rb'))
		val = pickle.load(open(dev, 'rb'))
		test = pickle.load(open(test, 'rb'))
		
		all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]    
		labels = list(set(all_labels))
		num_labels = len(labels)
		label_map = {i: label for i, label in enumerate(labels)} 
		# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
		pad_token_label_id = CrossEntropyLoss().ignore_index
		assert type(return_gm) == np.bool
		assert (type(loss_func_mask) == list) and len(loss_func_mask) == 7
		assert type(batch_size) == np.int or type(batch_size) == np.float
		assert type(lr_fm) == np.int or type(lr_fm) == np.float
		assert type(lr_gm) == np.int or type(lr_gm) == np.float
		assert type(use_accuracy_score) == np.bool
		assert type(n_epochs) == np.int or type(n_epochs) == np.float
		assert type(start_len) == np.int or type(start_len) == np.float
		assert type(stop_len) == np.int or type(stop_len) == np.float
		assert type(is_qt) == np.bool and type(is_qc) == np.bool
		assert (type(qt) == np.float and (qt >= 0 and qt <= 1)) or (type(qt) == np.ndarray and (np.all(np.logical_and(qt>=0, qt<=1)) ) )\
		 or (type(qt) == np.int and (qt == 0 or qt == 1))
		assert (type(qc) == np.float and (qc >= 0 and qc <= 1)) or (type(qc) == np.ndarray and (np.all(np.logical_and(qc>=0, qc<=1)) ) )\
		 or (type(qc) == np.int and (qc == 0 or qc == 1))
		assert metric_avg in ['micro', 'macro', 'samples', 'weighted', 'binary']

		batch_size_ = int(batch_size)
		n_epochs_ = int(n_epochs)
		start_len_ = int(start_len)
		stop_len_ = int(stop_len)

		score_used = "accuracy_score" if use_accuracy_score else "f1_score"

		assert start_len_ <= n_epochs_ and stop_len <= n_epochs_

		data_L = get_data(path_L, True, self.class_map)
		data_U = get_data(path_U, True, self.class_map)
		data_V = get_data(path_V, True, self.class_map)
		data_T = get_data(path_T, True, self.class_map)

		assert data_L[9] == self.n_classes and data_U[9] == data_L[9] and data_V[9] == data_L[9] and data_T[9] == data_L[9]

		x_sup = torch.tensor(data_L[0]).double() #0->1
		y_sup = torch.tensor(data_L[3]).long()        #3->4
		l_sup = torch.tensor(data_L[2]).long()
		s_sup = torch.tensor(data_L[6]).double()

		excluding = []
		temp_index = 0
		for temp in data_U[1]:
			if(np.all(temp == int(self.n_classes)) ):
				excluding.append(temp_index)
			temp_index+=1

		x_unsup = torch.tensor(np.delete(data_U[0], excluding, axis=0)).double()     #0->1
		y_unsup = torch.zeros((x_unsup).shape[0]).long()
		l_unsup = torch.tensor(np.delete(data_U[2], excluding, axis=0)).long()      
		s_unsup = torch.tensor(np.delete(data_U[6], excluding, axis=0)).double()

		x_valid = torch.tensor(data_V[0]).double()
		y_valid = data_V[3]
		l_valid = torch.tensor(data_V[2]).long()
		s_valid = torch.tensor(data_V[6]).double()

		x_test = torch.tensor(data_T[0]).double()
		y_test = data_T[3]
		l_test = torch.tensor(data_T[2]).long()
		s_test = torch.tensor(data_T[6]).double()

		y_sup = (y_sup).view(-1)
		y_valid = (y_valid).flatten()
		y_test = (y_test).flatten()
		
		z = torch.tensor(data_U[6], device = self.device).double() # continuous score

		assert self.n_features == x_sup.shape[1]
		assert self.n_lfs == l_sup.shape[1]
		if self. k == None:
			self.k = torch.tensor(data_L[8], device = self.device).long() # LF's classes
		else:
			assert torch.all(torch.tensor(data_L[8], device = self.device).double().eq(self.k))
		if self.continuous_mask == None:
			self.continuous_mask = torch.tensor(data_L[7], device = self.device).double() # Mask for s/continuous_mask
		else:
			assert torch.all(torch.tensor(data_L[7], device = self.device).double().eq(self.continuous_mask))

		assert np.all(data_L[8] == data_U[8]) and np.all(data_L[8] == data_V[8]) and np.all(data_L[8] == data_T[8])
		assert np.all(data_L[7] == data_U[7]) and np.all(data_L[7] == data_V[7]) and np.all(data_L[7] == data_T[7])

		assert x_sup.shape[1] == self.n_features and x_unsup.shape[1] == self.n_features \
		 and x_valid.shape[1] == self.n_features and x_test.shape[1] == self.n_features
		assert x_sup.shape[0] == y_sup.shape[0] and x_sup.shape[0] == l_sup.shape[0]\
		 and l_sup.shape == s_sup.shape and l_sup.shape[1] == self.n_lfs
		assert x_unsup.shape[0] == y_unsup.shape[0] and x_unsup.shape[0] == l_unsup.shape[0]\
		 and l_unsup.shape == s_unsup.shape and l_unsup.shape[1] == self.n_lfs
		assert x_valid.shape[0] == y_valid.shape[0] and x_valid.shape[0] == l_valid.shape[0]\
		 and l_valid.shape == s_valid.shape and l_valid.shape[1] == self.n_lfs
		assert x_test.shape[0] == y_test.shape[0] and x_test.shape[0] == l_test.shape[0]\
		 and l_test.shape == s_test.shape and l_test.shape[1] == self.n_lfs

		# clipping s
		s_sup[s_sup > 0.999] = 0.999
		s_sup[s_sup < 0.001] = 0.001
		s_unsup[s_unsup > 0.999] = 0.999
		s_unsup[s_unsup < 0.001] = 0.001
		s_valid[s_valid > 0.999] = 0.999
		s_valid[s_valid < 0.001] = 0.001
		s_test[s_test > 0.999] = 0.999
		s_test[s_test < 0.001] = 0.001
		z[z > 0.999] = 0.999 # clip s
		z[z < 0.001] = 0.001 # clip s

		l = torch.cat([l_sup, l_unsup])
		s = torch.cat([s_sup, s_unsup])
		x_train = torch.cat([x_sup, x_unsup])
		y_train = torch.cat([y_sup, y_unsup])
		supervised_mask = torch.cat([torch.ones(l_sup.shape[0]), torch.zeros(l_unsup.shape[0])])

		if is_qt:
			qt_ = torch.tensor(qt, device = self.device).double() if type(qt) == np.ndarray else (torch.ones(self.n_lfs, device = self.device).double() * qt)
		else: 
			prec_lfs=[]
			for i in range(self.n_lfs):
				correct = 0
				for j in range(len(y_valid)):
					if y_valid[j] == l_valid[j][i]:
						correct+=1
				prec_lfs.append(correct/len(y_valid))
			qt_ = torch.tensor(prec_lfs).double()

		if is_qc:
			qc_ = torch.tensor(qc, device = self.device).double() if type(qc) == np.ndarray else qc
		else:
			qc_ = torch.tensor(np.mean(s_valid, axis = 0), device = self.device)

		file = None
		# if path_log != None:
		# 	file = open(path_log, "a+")
		# 	file.write("JL log:\tn_classes: {}\tn_LFs: {}\tn_features: {}\tn_hidden: {}\tfeature_model:{}\tlr_fm: {}\tlr_gm:{}\tuse_accuracy_score: {}\tn_epochs:{}\tstart_len: {}\tstop_len:{}\n".format(\
		# 		self.n_classes, self.n_lfs, self.n_features, self.n_hidden, self.feature_based_model, lr_fm, lr_gm, use_accuracy_score, n_epochs, start_len, stop_len))
		# else:
		# 	print("JL log:\tn_classes: {}\tn_LFs: {}\tn_features: {}\tn_hidden: {}\tfeature_model:{}\tlr_fm: {}\tlr_gm:{}\tuse_accuracy_score: {}\tn_epochs:{}\tstart_len: {}\tstop_len:{}".format(\
		# 		self.n_classes, self.n_lfs, self.n_features, self.n_hidden, self.feature_based_model, lr_fm, lr_gm, use_accuracy_score, n_epochs, start_len, stop_len))
        ############################################################################################################################################
		


		#############################################################################################################################################
		#Algo starting
		optimizer_fm = torch.optim.AdamW(self.feature_model.parameters(), lr=lr_fm, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, maximize=False, foreach=None, capturable=False)
		# optimizer_fm = torch.optim.Adam(self.feature_model.parameters(), lr=lr_fm)
# 		scheduler = torch.optim.lr_scheduler.StepLR(optimizer_fm, step_size=10, gamma=0.1)
        ######################################################################
		pad_token_label_id=CrossEntropyLoss().ignore_index
		##
		# print(train_u[3])
		#TODO Make CordDataset to x
		train_u_dataset = CordDataset(train_u, tokenizer, labels, pad_token_label_id,self.n_lfs)
		train_u_sampler = RandomSampler(train_u_dataset)
		train_u_dataloader = DataLoader(train_u_dataset,
						sampler=train_u_sampler,
						batch_size=2)

		train_dataset = CordDataset(train, tokenizer, labels, pad_token_label_id,self.n_lfs)
		train_sampler = RandomSampler(train_dataset)
		optimizer_gm = torch.optim.Adam([self.theta, self.pi], lr = lr_gm, weight_decay=0)	
		# dataset = TensorDataset(x_train, y_train, l, s, supervised_mask)
		# loader = DataLoader(dataset, batch_size = batch_size_, shuffle = True, drop_last = False, pin_memory = True)
		#########################################################################
		test_dataset = CordDataset(test, tokenizer, labels, pad_token_label_id,self.n_lfs)
		test_sampler = SequentialSampler(test_dataset)
		test_dataloader = DataLoader(test_dataset,sampler=test_sampler,batch_size=2)
		##########################################################################
		eval_dataset = CordDataset(val, tokenizer, labels, pad_token_label_id,self.n_lfs)
		eval_sampler = SequentialSampler(eval_dataset)
		eval_dataloader = DataLoader(eval_dataset,sampler=eval_sampler,batch_size=2)
		##########################################################################
		
		train_dataloader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=2)

       ############################################################################################
		best_score_fm_test, best_score_gm_test, best_epoch, best_score_fm_val, best_score_gm_val = 0,0,0,0,0
		best_prec_fm_test, best_recall_fm_test, best_prec_gm_test, best_recall_gm_test= 0,0,0,0

		gm_test_acc, fm_test_acc = -1, -1

		stopped_epoch = -1
		stop_early_fm, stop_early_gm = [], []
		# supervised_criterion = torch.nn.functional.cross_entropy
		optimizer = AdamW(self.feature_model.parameters(), lr=5e-5)
		i=0
		for batch in test_dataloader:
			i=i+1
			if(i==1): 
				l_testz=batch[5].to(self.device)
			else:
				l_testz=torch.cat([l_testz,batch[5].to(self.device)])
		# print(l_test)
		j=0
		for batch in eval_dataloader:
			j=j+1
			if(j==1): 
				l_validz=batch[5].to(self.device)
			else:
				l_validz=torch.cat([l_validz,batch[5].to(self.device)])	
		# print(l_valid)
		k=0
		for batch in train_dataloader:
			k=k+1
			if(k==1): 
				l_tra=batch[5].to(self.device)
			else:
				l_tra=torch.cat([l_tra,batch[5].to(self.device)])
		q=0
		for batch in train_dataloader:
			q=q+1
			if(q==1): 
				l_tra_u=batch[5].to(self.device)
			else:
				l_tra_u=torch.cat([l_tra_u,batch[5].to(self.device)])				

		with tqdm(total=n_epochs_) as pbar:
			# self.feature_model.train()
			global_step = 0
			global_step_i = 0
			for epoch in range(n_epochs_):
				dataloader_iterator = iter(train_dataloader)
				for batchz in tqdm(train_u_dataloader):
					try:
						batch = next(dataloader_iterator)
					except StopIteration: 
						dataloader_iterator = iter(train_dataloader)
						batch = next(dataloader_iterator) 
					optimizer_fm.zero_grad()
					optimizer_gm.zero_grad()					

					# supervised_indices = sample[4].nonzero().view(-1)
					# unsupervised_indices = (1-sample[4]).nonzero().squeeze()
					device = self.device
					input_ids = batch[0].to(device)
					bbox = batch[4].to(device)
					attention_mask = batch[1].to(device)
					token_type_ids = batch[2].to(device)
					labels = batch[3].to(device)
					l_sz = batch[5].to(device)
					# s_sz = l_sz
					# s_sz[s_sz!=0] = 1
					s_sz = torch.zeros(2,512,self.n_lfs).to(device)
					s_sz[s_sz < 0.001] = 0.001
					# forward pass
					outputs = self.feature_model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
									labels=labels)
					output=outputs[1]
					loss_1= outputs.loss
					# outputs = self.feature_model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
					# 				labels=labels)[1]
					
					input_idsz = batchz[0].to(device)
					bboxz = batchz[4].to(device)
					l=batchz[5].to(device)
					# s=l
					# s[s!=0] = 1
					s = torch.zeros(2,512,self.n_lfs).to(device)
					s[s < 0.001] = 0.001
					attention_maskz = batchz[1].to(device)
					token_type_idsz = batchz[2].to(device)
					q=self.feature_model(input_ids=input_idsz, bbox=bboxz, attention_mask=attention_maskz, token_type_ids=token_type_idsz)[0]
					if(loss_func_mask[1]):
						unsupervised_fm_probability = torch.nn.Softmax(dim = 1)(q)    
						loss_2 = entropy(unsupervised_fm_probability)
					else:
						loss_2 = 0
					
					if(loss_func_mask[2]):
						y_pred_unsupervised=np.zeros((2,512,1))
						for i in range(2):
							y_pre = predict_gm_labels(self.theta, self.pi,l[i],s[i], self.k, self.n_classes, self.continuous_mask, qc_, self.device)
							y_pred_unsupervised[i,:,0]=y_pre
						y_pred_unsupervised=torch.cat([torch.tensor(y_pred_unsupervised[0]),torch.tensor(y_pred_unsupervised[1])])
						y_pred_unsupervised=torch.squeeze(y_pred_unsupervised,1)
						y_pred_unsupervised = y_pred_unsupervised.type(torch.LongTensor)	
						q=q.cpu().detach().numpy()
						q=torch.tensor(q,device = self.device)
						q_=torch.cat([q[0],q[1]]) 		 
						loss_3 = torch.nn.functional.cross_entropy(q_, torch.tensor(y_pred_unsupervised, device = self.device))
					else:
						loss_3 = 0
						
					if (loss_func_mask[3]):
						loss_arr=[]
						for i in range(2):
							loss4 =  log_likelihood_loss_supervised(self.theta, self.pi, x_train, l_sz[i],s_sz[i], self.k, self.n_classes, self.continuous_mask, qc_, self.device)
							loss_arr.append(loss4)
						loss_4=(loss_arr[0]+loss_arr[1])/2
					else:
						loss_4 = 0
					loss_arr=[]
					if(loss_func_mask[4]):
						for i in range(2):
							loss5 = log_likelihood_loss(self.theta, self.pi,l[i],s[i], self.k, self.n_classes, self.continuous_mask, qc_, self.device)
							loss_arr.append(loss5)
						loss_5=(loss_arr[0]+loss_arr[1])/2
					else:
						loss_5 = 0

					if(loss_func_mask[5]):
						# if(len(supervised_indices) >0):
						# supervised_indices = supervised_indices.tolist()
						# outputs=outputs.cpu().detach().numpy()
						# outputs=torch.tensor(outputs,device = self.device)
						loss1 = probability(self.theta, self.pi, l[0],s[0], self.k, self.n_classes, self.continuous_mask, qc_, self.device)
						probs_graphical_1 = (loss1.t() / (loss1.sum(1)+1e-15)).t()
						# fm_1=torch.cat([outputs[0],q[0]])
						probs_fm_1 = torch.nn.Softmax(dim = 1)(q[0])
						loss_6_1 = kl_divergence(probs_fm_1, probs_graphical_1)

						loss2 = probability(self.theta, self.pi, l[1],s[1], self.k, self.n_classes, self.continuous_mask, qc_, self.device)
						probs_graphical_2 = (loss2.t() / (loss2.sum(1)+1e-15)).t()
						# fm_2=torch.cat([outputs[1],q[1]])
						probs_fm_2 = torch.nn.Softmax(dim = 1)(q[1])
						loss_6_2 = kl_divergence(probs_fm_2, probs_graphical_2)
						loss_6=(loss_6_1+loss_6_2)/2	
					else:
						loss_6 = 0
	# loss1 = probability(self.theta, self.pi, torch.cat([l_sz[0], l[0]]),torch.cat([s_sz[0],s[0]]), self.k, self.n_classes, self.continuous_mask, qc_, self.device)
	# 					probs_graphical_1 = (loss1.t() / loss1.sum(1)).t()
	# 					fm_1=torch.cat([outputs[0],q[0]])
	# 					probs_fm_1 = torch.nn.Softmax(dim = 1)(fm_1)
	# 					loss_6_1 = kl_divergence(probs_fm_1, probs_graphical_1)
						
	# 					loss2 = probability(self.theta, self.pi, torch.cat([l_sz[1], l[1]]),torch.cat([s_sz[1],s[1]]), self.k, self.n_classes, self.continuous_mask, qc_, self.device)
	# 					probs_graphical_2 = (loss2.t() / loss2.sum(1)).t()
	# 					fm_2=torch.cat([outputs[1],q[1]])
	# 					probs_fm_2 = torch.nn.Softmax(dim = 1)(fm_2)
	# 					loss_6_2 = kl_divergence(probs_fm_2, probs_graphical_2)
	# 					loss_6=(loss_6_1+loss_6_2)/2	
	# 				else:
	# 					loss_6 = 0
					if(loss_func_mask[6]):
						prec_loss = precision_loss(self.theta, self.k, self.n_classes, qt_, self.device)
					else:
						prec_loss = 0
					loss_ = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + prec_loss
					if global_step_i % 100 == 0:
						# print(f"Loss in feature model after {global_step} steps: {loss_1.item()}")
						print(f"Loss_1 after {global_step_i} steps: {loss_1}")
						print(f"Loss_2 after {global_step_i} steps: {loss_2}")
						print(f"Loss_3 after {global_step_i} steps: {loss_3}")
						print(f"Loss_4 after {global_step_i} steps: {loss_4}")
						print(f"Loss_5 after {global_step_i} steps: {loss_5}")
						print(f"Loss_6 after {global_step_i} steps: {loss_6}")
						print(f"Loss_7 after {global_step_i} steps: {prec_loss}")
						print(f"Loss in geneartive model after {global_step_i} steps: {loss_}")
						if path_log != None:
							file = open(path_log, "a+")
							file.write("Loss_1 after: {}\t steps: {}\n".format(global_step_i, loss_1))
							file.write("Loss_2 after: {}\t steps: {}\n".format(global_step_i, loss_2))
							file.write("Loss_3 after: {}\t steps: {}\n".format(global_step_i, loss_3))
							file.write("Loss_4 after: {}\t steps: {}\n".format(global_step_i, loss_4))
							file.write("Loss_5 after: {}\t steps: {}\n".format(global_step_i, loss_5))
							file.write("Loss_6 after: {}\t steps: {}\n".format(global_step_i, loss_6))
							file.write("Loss_7 after: {}\t steps: {}\n".format(global_step_i, prec_loss))
							file.write("total_Loss after: {}\t steps: {}\n".format(global_step_i, loss_))
							file.close() 
					if loss_ != 0:
						writer.add_scalar("Loss/loss_1", loss_1, epoch)
						writer.add_scalar("Loss/loss_2", loss_2, epoch)
						writer.add_scalar("Loss/loss_3", loss_3, epoch)
						writer.add_scalar("Loss/loss_4", loss_4, epoch)
						writer.add_scalar("Loss/loss_5", loss_5, epoch)
						writer.add_scalar("Loss/prec_loss", prec_loss, epoch)
						writer.add_scalar("Loss/total_loss", loss_, epoch)
						loss_.backward()
						optimizer_fm.step()
						optimizer_gm.step()
	# 						scheduler.step()
					global_step_i += 1
				
				############################################################################################################################
			


				#################################################################################### 
				# s_test = torch.zeros(51200,7).to(device)
				# s_test[s_test < 0.001] = 0.001
				# s_valid = torch.zeros(51200,7).to(device)
				# s_valid[s_valid < 0.001] = 0.001

				
				###################################################################################
				



                ###########################################################################################################################

				# i=0	
				# for i in range(99):
				# 	l_test=torch.cat([l_testz[i],l_testz[i+1]])

				# i=0	
				# for i in range(99):
				# 	l_test=torch.cat([l_validz[i],l_validz[i+1]])	


				############################################################################################################################			 
                #gm test
				y_pred = predict_gm_labels(self.theta, self.pi, l_test.to(device = self.device), s_test.to(device = self.device), self.k, self.n_classes, self.continuous_mask, qc_, self.device)
				print(y_pred)
				from sklearn.metrics import accuracy_score
				from sklearn.metrics import f1_score
				from sklearn.metrics import precision_score as prec_score
				from sklearn.metrics import recall_score as recall_score
				if use_accuracy_score:
					gm_test_acc = accuracy_score(y_test, y_pred)
				else:
					gm_test_acc = f1_score(y_test, y_pred, average = metric_avg)
				gm_test_prec = prec_score(y_test, y_pred, average = metric_avg)
				gm_test_recall = recall_score(y_test, y_pred, average = metric_avg)

				#gm validation
				y_pred = predict_gm_labels(self.theta, self.pi, l_valid.to(device = self.device), s_valid.to(device = self.device), self.k, self.n_classes, self.continuous_mask, qc_, self.device)
				if use_accuracy_score:
					gm_valid_acc = accuracy_score(y_valid, y_pred)
				else:
					gm_valid_acc = f1_score(y_valid, y_pred, average = metric_avg)
				
								#fm test
				#############################################################
				
				nb_test_steps=0
				test_loss=0.0
				preds=None
				for batch in tqdm(test_dataloader, desc="Evaluating"):
					with torch.no_grad():
						input_ids = batch[0].to(device)
						bbox = batch[4].to(device)
						attention_mask = batch[1].to(device)
						token_type_ids = batch[2].to(device)
						labels = batch[3].to(device)
						z=batch[5].to(device)
						# forward pass
						outputs = self.feature_model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
										labels=labels)
						# get the loss and logits
						tmp_test_loss = outputs.loss
						logits = outputs.logits

						test_loss += tmp_test_loss.item()
						nb_test_steps += 1

						# compute the predictions
						if preds is None:
							preds = logits.detach().cpu().numpy()
							out_label_ids = labels.detach().cpu().numpy()
						else:
							preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
							out_label_ids = np.append(
								out_label_ids, labels.detach().cpu().numpy(), axis=0
							)

				# compute average evaluation loss
				test_loss = test_loss / nb_test_steps
				preds = np.argmax(preds, axis=2)

				out_label_list = [[] for _ in range(out_label_ids.shape[0])]
				preds_list = [[] for _ in range(out_label_ids.shape[0])]
				# label_map = {i: label for i, label in enumerate(labels)}
				for i in range(out_label_ids.shape[0]):
					for j in range(out_label_ids.shape[1]):
						if out_label_ids[i, j] != pad_token_label_id:
							out_label_list[i].append(label_map[out_label_ids[i][j]])
							preds_list[i].append(label_map[preds[i][j]])
				####################################################################################
				from seqeval.metrics import classification_report,precision_score
				from seqeval.metrics import f1_score as f1
				from seqeval.metrics import recall_score as rec
				from seqeval.metrics import accuracy_score as acc
				if use_accuracy_score:
					fm_test_acc = acc(out_label_list, preds_list)
				else:
					fm_test_acc = acc(out_label_list, preds_list)
				fm_test_prec = precision_score(out_label_list, preds_list)
				fm_test_recall = rec(out_label_list, preds_list)
				print("test accuracy -",fm_test_acc)
				print("precision accuracy -",fm_test_prec)
				print("recall accuracy -",fm_test_recall)
				writer.add_scalar('Accuracy/test',fm_test_acc , epoch)
				if path_log != None:
					file = open(path_log, "a+")
					file.write("epoch: {}\tfm_test_acc: {}\n".format(epoch, fm_test_acc))
					file.close() 
                ##################################################################################		  
				#fm validation
                ###################################################################################################
                ##################################################################################
				###################################################################################
				nb_eval_steps=0
				eval_loss=0.0
				preds=None
				for batch in tqdm(eval_dataloader, desc="Evaluating"):
					with torch.no_grad():
						input_ids = batch[0].to(device)
						bbox = batch[4].to(device)
						attention_mask = batch[1].to(device)
						token_type_ids = batch[2].to(device)
						labels = batch[3].to(device)

						# forward pass
						outputs = self.feature_model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
										labels=labels)
						# get the loss and logits
						tmp_eval_loss = outputs.loss
						logits = outputs.logits

						eval_loss += tmp_eval_loss.item()
						nb_eval_steps += 1

						# compute the predictions
						if preds is None:
							preds = logits.detach().cpu().numpy()
							out_label_ids = labels.detach().cpu().numpy()
						else:
							preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
							out_label_ids = np.append(
								out_label_ids, labels.detach().cpu().numpy(), axis=0
							)

				# compute average evaluation loss
				eval_loss = eval_loss / nb_eval_steps
				predsz = np.argmax(preds, axis=2)

				out_label_listz = [[] for _ in range(out_label_ids.shape[0])]
				preds_listz = [[] for _ in range(out_label_ids.shape[0])]

				for i in range(out_label_ids.shape[0]):
					for j in range(out_label_ids.shape[1]):
						if out_label_ids[i, j] != pad_token_label_id:
							out_label_listz[i].append(label_map[out_label_ids[i][j]])
							preds_listz[i].append(label_map[predsz[i][j]])

				###################################################################################################
				from seqeval.metrics import accuracy_score as acc
				from seqeval.metrics import classification_report,precision_score
				from seqeval.metrics import f1_score as f1
				from seqeval.metrics import recall_score as rec
				if use_accuracy_score:
					fm_valid_acc = acc(out_label_listz, preds_listz)
					print("fm_validation",fm_valid_acc)
				else:
					fm_valid_acc = acc(out_label_listz, preds_listz)
					fm_val_prec = precision_score(out_label_listz, preds_listz)
					fm_val_recall = rec(out_label_listz, preds_listz)
					print("fm_validation",fm_valid_acc)
					print("precision val accuracy -",fm_val_prec)
					print("recall val accuracy -",fm_val_recall)
					writer.add_scalar('Accuracy/validation',fm_valid_acc , epoch)
				if path_log != None:
					file = open(path_log, "a+")
					file.write("epoch: {}\tfm_val_acc: {}\n".format(epoch, fm_valid_acc))
				file.close() 
				(self.feature_model).train()

				# if path_log != None:
				# 	file.write("{}: Epoch: {}\tgm_valid_score: {}\tfm_valid_score: {}\n".format(score_used, epoch, gm_valid_acc, fm_valid_acc))
				# 	if epoch % 5 == 0:
				# 		file.write("{}: Epoch: {}\tgm_test_score: {}\tfm_test_score: {}\n".format(score_used, epoch, gm_test_acc, fm_test_acc))
				# else:
				# 	print("{}: Epoch: {}\tgm_valid_score: {}\tfm_valid_score: {}".format(score_used, epoch, gm_valid_acc, fm_valid_acc))
				# 	if epoch % 5 == 0:
				# 		print("{}: Epoch: {}\tgm_test_score: {}\tfm_test_score: {}".format(score_used, epoch, gm_test_acc, fm_test_acc))
                # epoch > start_len_ 
				if gm_valid_acc >= best_score_gm_val and gm_valid_acc >= best_score_fm_val:
					if gm_valid_acc == best_score_gm_val or gm_valid_acc == best_score_fm_val:
						if best_score_gm_test < gm_test_acc or best_score_fm_test < fm_test_acc:
							best_epoch = epoch
							self.pi_optimal = (self.pi).detach().clone()
							self.theta_optimal = (self.theta).detach().clone()
							self.fm_optimal_params = deepcopy((self.feature_model).state_dict())

							best_score_fm_val = fm_valid_acc
							best_score_fm_test = fm_test_acc
							best_score_gm_val = gm_valid_acc
							best_score_gm_test = gm_test_acc

							best_prec_fm_test = fm_test_prec
							best_recall_fm_test  = fm_test_recall
							best_prec_gm_test = gm_test_prec
							best_recall_gm_test  = gm_test_recall
					else:
						best_epoch = epoch
						self.pi_optimal = (self.pi).detach().clone()
						self.theta_optimal = (self.theta).detach().clone()
						self.fm_optimal_params = deepcopy((self.feature_model).state_dict())

						best_score_fm_val = fm_valid_acc
						best_score_fm_test = fm_test_acc
						best_score_gm_val = gm_valid_acc
						best_score_gm_test = gm_test_acc

						best_prec_fm_test = fm_test_prec
						best_recall_fm_test  = fm_test_recall
						best_prec_gm_test = gm_test_prec
						best_recall_gm_test  = gm_test_recall
						stop_early_fm = []
						stop_early_gm = []
                # epoch > start_len_
				if fm_valid_acc >= best_score_fm_val and fm_valid_acc >= best_score_gm_val:
					if fm_valid_acc == best_score_fm_val or fm_valid_acc == best_score_gm_val:
						if best_score_fm_test < fm_test_acc or best_score_gm_test < gm_test_acc:
							best_epoch = epoch
							self.pi_optimal = (self.pi).detach().clone()
							self.theta_optimal = (self.theta).detach().clone()
							self.fm_optimal_params = deepcopy((self.feature_model).state_dict())

							best_score_fm_val = fm_valid_acc
							best_score_fm_test = fm_test_acc
							best_score_gm_val = gm_valid_acc
							best_score_gm_test = gm_test_acc

							best_prec_fm_test = fm_test_prec
							best_recall_fm_test  = fm_test_recall
							best_prec_gm_test = gm_test_prec
							best_recall_gm_test  = gm_test_recall
					else:
						best_epoch = epoch
						self.pi_optimal = (self.pi).detach().clone()
						self.theta_optimal = (self.theta).detach().clone()
						self.fm_optimal_params = deepcopy((self.feature_model).state_dict())
						
						best_score_fm_val = fm_valid_acc
						best_score_fm_test = fm_test_acc
						best_score_gm_val = gm_valid_acc
						best_score_gm_test = gm_test_acc

						best_prec_fm_test = fm_test_prec
						best_recall_fm_test  = fm_test_recall
						best_prec_gm_test = gm_test_prec
						best_recall_gm_test  = gm_test_recall
						stop_early_fm = []
						stop_early_gm = []

				if len(stop_early_fm) > stop_len_ and len(stop_early_gm) > stop_len_ and (all(best_score_fm_val >= k for k in stop_early_fm) or \
				all(best_score_gm_val >= k for k in stop_early_gm)):
					stopped_epoch = epoch
					break
				else:
					stop_early_fm.append(fm_valid_acc)
					stop_early_gm.append(gm_valid_acc)

				pbar.update()
				#epoch for loop ended
		writer.close()
		if stopped_epoch == -1:
			print('best_epoch: {}'.format(best_epoch))
		else:
			print('early stopping at epoch: {}\tbest_epoch: {}'.format(stopped_epoch, best_epoch))

		if use_accuracy_score:
			print('score used: accuracy_score')
		else:
			print('score used: f1_score')
		print('best_gm_val_score:{}\tbest_fm_val_score:{}'.format(\
			best_score_gm_val, best_score_fm_val))	
		print('best_gm_test_score:{}\tbest_fm_test_score:{}'.format(\
			best_score_gm_test, best_score_fm_test))
		print('best_gm_test_precision:{}\tbest_fm_test_precision:{}'.format(\
			best_prec_gm_test, best_prec_fm_test))
		print('best_gm_test_recall:{}\tbest_fm_test_recall:{}'.format(\
			best_recall_gm_test, best_recall_fm_test))
			
		# Algo ended
        
		# below prints and writes to file, the final test accuracies
		print("final_gm_test_acc: {}\tfinal_fm_test_acc: {}\n".format(gm_test_acc, fm_test_acc))
		

		(self.feature_model).load_state_dict(self.fm_optimal_params)
		
		pad_token_label_id=CrossEntropyLoss().ignore_index
		eval_loss = 0.0
		nb_eval_steps = 0
		preds = None
		out_label_ids = None
		#################################################################################################################################
		# q=0
		# for batch in (tqdm(train_u_dataloader)):
		# 	q=q+1
		# 	if(q==1):
		# 		input_ids=batch[0].to(device)
		# 		attention_mask=batch[1].to(device)
		# 		token_type_ids =batch[2].to(device)
		# 		labels=batch[3].to(device)
		# 		bbox =batch[4].to(device)
		# 	else:
		# 		input_ids=torch.cat((batch[0].to(device),input_ids))
		# 		attention_mask=torch.cat((batch[1].to(device),attention_mask))
		# 		token_type_ids =torch.cat((batch[2].to(device),token_type_ids))
		# 		labels=torch.cat((batch[3].to(device),labels))
		# 		bbox =torch.cat((batch[4].to(device),bbox))

		#################################################################################################################################
		# put model in evaluation mode
		# (self.feature_model).eval()
		# preds=torch.nn.Softmax(dim = 1)(self.feature_model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)[0])
		# compute average evaluation loss
		
		# put model in evaluation mode
		(self.feature_model).eval()
		for batch in tqdm(train_u_dataloader, desc="training"):
			with torch.no_grad():
				input_ids = batch[0].to(device)
				bbox = batch[4].to(device)
				attention_mask = batch[1].to(device)
				token_type_ids = batch[2].to(device)
				labels = batch[3].to(device)

				# forward pass
				outputs = self.feature_model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
								labels=labels)
				# get the loss and logits
				tmp_eval_loss = outputs.loss
				logits = outputs.logits

				eval_loss += tmp_eval_loss.item()
				nb_eval_steps += 1

				# compute the predictions
				if preds is None:
					preds = logits.detach().cpu().numpy()
					out_label_ids = labels.detach().cpu().numpy()
				else:
					preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
					out_label_ids = np.append(
						out_label_ids, labels.detach().cpu().numpy(), axis=0
					)

		# compute average evaluation loss
		eval_loss = eval_loss / nb_eval_steps
		(self.feature_model).train()

		if return_gm:
			return preds, (probability(self.theta_optimal, self.pi_optimal, torch.abs(torch.tensor(data_U[2], device = self.device).long()), z, \
				self.k, self.n_classes, self.continuous_mask, qc_, self.device)).cpu().detach().numpy()
		else:
			return preds

	def fit_and_predict(self, path_L, path_U, path_V, path_T, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, path_log = None, return_gm = False, n_epochs = 100, start_len = 7,\
	 stop_len = 10, is_qt = True, is_qc = True, qt = 0.9, qc = 0.85, metric_avg = 'binary', need_strings = False):
		'''
		Args:
			path_L: Path to pickle file of labelled instances
			path_U: Path to pickle file of unlabelled instances
			path_V: Path to pickle file of validation instances
			path_T: Path to pickle file of test instances
			loss_func_mask: list of size 7 where loss_func_mask[i] should be 1 if Loss function (i+1) should be included, 0 else. Checkout Eq(3) in :cite:p:`DBLP:journals/corr/abs-2008-09887`
			batch_size: Batch size, type should be integer
			lr_fm: Learning rate for feature model, type is integer or float
			lr_gm: Learning rate for graphical model(cage algorithm), type is integer or float
			use_accuracy_score: The score to use for termination condition on validation set. True for accuracy_score, False for f1_score
			path_log: Path to log file to append log. Default is None which prints accuracies/f1_scores is printed to terminal
			return_gm: Return the predictions of graphical model? the allowed values are True, False. Default value is False
			n_epochs: Number of epochs in each run, type is integer, default is 100
			start_len: A parameter used in validation, refers to the least epoch after which validation checks need to be performed, type is integer, default is 7
			stop_len: A parameter used in validation, refers to the least number of continuous epochs of non incresing validation accuracy after which the training should be stopped, type is integer, default is 10
			is_qt: True if quality guide is available(and will be provided in 'qt' argument). False if quality guide is intended to be found from validation instances. Default is True
			is_qc: True if quality index is available(and will be provided in 'qc' argument). False if quality index is intended to be found from validation instances. Default is True
			qt: Quality guide of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.9
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			metric_avg: Average metric to be used in calculating f1_score/precision/recall, default is 'binary'
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False
		Return:
			If return_gm is True; the return value is two predicted labels of numpy array of shape (num_instances, ), first one is through feature model, other one through graphical model.
			Else; the return value is predicted labels of numpy array of shape (num_instances,) through feature model. It is suggested to use the probailities of feature model
		'''
		assert type(need_strings) == np.bool
		if return_gm:
			proba_1, proba_2 = self.fit_and_predict_proba(path_L, path_U, path_V, path_T, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, path_log, return_gm, n_epochs, start_len,\
	 		stop_len, is_qt, is_qc, qt, qc, metric_avg)
			return get_predictions(proba_1, self.class_map, self.class_dict, need_strings), get_predictions(proba_2, self.class_map, self.class_dict, need_strings)
		else:
			proba = self.fit_and_predict_proba(path_L, path_U, path_V, path_T, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, path_log, return_gm, n_epochs, start_len,\
	 		stop_len, is_qt, is_qc, qt, qc, metric_avg)
			return get_predictions(proba, self.class_map, self.class_dict, need_strings)

	
	def predict_gm_proba(self, path_test, qc = 0.85):
		'''
			Used to find the predicted labels based on the trained parameters of graphical model(CAGE)
		Args:
			path_test: Path to the pickle file containing test data set
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
		
		Return:
			numpy.ndarray of shape (num_instances, num_classes) where i,j-th element is the probability of ith instance being the jth class(the jth value when sorted in ascending order of values in Enum)
			[Note: no aggregration/algorithm-running will be done using the current input]. It is suggested to use the probailities of feature model
		'''
		assert (type(qc) == np.float and (qc >= 0 and qc <= 1)) or (type(qc) == np.ndarray and (np.all(np.logical_and(qc>=0, qc<=1)) ) )\
		 or (type(qc) == np.int and (qc == 0 or qc == 1))

		data = get_data(path_test, True, self.class_map)
		s_test = torch.tensor(data[6], device = self.device).double()
		s_test[s_test > 0.999] = 0.999
		s_test[s_test < 0.001] = 0.001
		assert (data[2]).shape[1] == self.n_lfs and data[9] == self.n_classes
		assert (data[0].shape)[1] == self.n_features
		temp_k = torch.tensor(data[8], device = self.device).long()
		assert self.k == None or torch.all(temp_k.eq(self.k))
		temp_continuous_mask = torch.tensor(data[7], device = self.device).double()
		assert self.continuous_mask == None or torch.all(temp_continuous_mask.eq(self.continuous_mask))
		m_test = torch.abs(torch.tensor(data[2], device = self.device).long())
		qc_ = torch.tensor(qc, device = self.device).double() if type(qc) == np.ndarray else qc

		if self.continuous_mask == None or self.k == None:
			print("Warning: Predict is used before training any paramters in JL class. Hope you have loaded parameters.")
		return (probability(self.theta_optimal, self.pi_optimal, m_test, s_test, temp_k, self.n_classes, temp_continuous_mask, qc_, self.device)).cpu().detach().numpy()
		
	def predict_fm_proba(self, x_test):
		'''
			Used to find the predicted labels based on the trained parameters of feature model
		Args:
			x_test: numpy array of shape (num_instances, num_features) containing data whose labels are to be predicted
		
		Return:
			numpy.ndarray of shape (num_instances, num_classes) where i,j-th element is the probability of ith instance being the jth class(the jth value when sorted in ascending order of values in Enum)
			[Note: no aggregration/algorithm-running will be done using the current input]. It is suggested to use the probailities of feature model
		'''
		# assert x_test.shape[1] == self.n_features

		if self.continuous_mask == None or self.k == None:
			print("Warning: Predict is used before training any paramters in JL class. Hope you have loaded parameters.")

		(self.feature_model).load_state_dict(self.fm_optimal_params)
		device = self.device
		(self.feature_model).eval()
		pad_token_label_id=CrossEntropyLoss().ignore_index
		###############################################################################################################################
		# fm_predictions = (torch.nn.Softmax(dim = 1)(self.feature_model(torch.tensor(x_test, device = self.device).double()))).cpu().detach().numpy()
		labels = ["PRICE", "QUANTITY", "MENU"]
		test_dataset = CordDataset(x_test, tokenizer, labels, pad_token_label_id)
		test_sampler = SequentialSampler(test_dataset)
		test_dataloader = DataLoader(test_dataset,sampler=test_sampler,batch_size=2)
		##################################################################################
		###################################################################################
		nb_eval_steps=0
		preds=None
		for batch in tqdm(test_dataloader, desc="Evaluating"):
			with torch.no_grad():
				input_ids = batch[0].to(device)
				bbox = batch[4].to(device)
				attention_mask = batch[1].to(device)
				token_type_ids = batch[2].to(device)
				labels = batch[3].to(device)

				# forward pass
				outputs = self.feature_model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
								labels=labels)
				# get the loss and logits
				tmp_eval_loss = outputs.loss
				logits = outputs.logits

				eval_loss += tmp_eval_loss.item()
				nb_eval_steps += 1

				# compute the predictions
				if preds is None:
					preds = logits.detach().cpu().numpy()
					out_label_ids = labels.detach().cpu().numpy()
				else:
					preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
					out_label_ids = np.append(
						out_label_ids, labels.detach().cpu().numpy(), axis=0
					)

		# compute average evaluation loss
		test_loss = test_loss / nb_eval_steps
		preds = np.argmax(preds, axis=2)

        ################################################################################################################################ 
		(self.feature_model).train()

		return preds

	def predict_gm(self, path_test, qc = 0.85, need_strings = False):
		'''
			Used to find the predicted labels based on the trained parameters of graphical model(CAGE)
		Args:
			path_test: Path to the pickle file containing test data set
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False
		
		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels. Elements are numbers/strings depending on need_strings attribute is false/true resp.
			[Note: no aggregration/algorithm-running will be done using the current input]. It is suggested to use the probailities of feature model
		'''
		assert type(need_strings) == np.bool
		return get_predictions(self.predict_gm_proba(path_test, qc), self.class_map, self.class_dict, need_strings)

	def predict_fm(self, x_test, need_strings = False):
		'''
			Used to find the predicted labels based on the trained parameters of feature model
		Args:
			x_test: numpy array of shape (num_instances, num_features) containing data whose labels are to be predicted
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False
		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels. Elements are numbers/strings depending on need_strings attribute is false/true resp.
			[Note: no aggregration/algorithm-running will be done using the current input]. It is suggested to use the probailities of feature model
		'''
		assert type(need_strings) == np.bool
		return get_predictions(self.predict_fm_proba(x_test), self.class_map, self.class_dict, need_strings)