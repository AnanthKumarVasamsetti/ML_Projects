import glob
import os
import numpy as np
from collections import defaultdict
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
emails = []
labels = [] #1 for Spam & 0 for Ham

def letters_only(astr):
    return astr.isalpha()

def clean_text(docs):
    all_words = set(names.words())
    lemmatizer = WordNetLemmatizer()
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower()) for word in doc.split() if letters_only(word) and word not in all_words]))
    return cleaned_docs

#Function to get indices of emails as {0:[NS1,NS2,....,NSn],1:[S1,S2,....,Sm]}
#Here training samples indices are grouped by classes
def get_label_index(labels):
    label_index = defaultdict(list)
    for index,labels in enumerate(labels):
        label_index[labels].append(index)
    return label_index
"""
Compute prior based on training samples
Args:
    label_index (grouped sample indices by class)
Returns:
    dictionary, with class label as key, corresponding
    prior as the value
"""
def get_prior(label_index):
    prior = {label : len(index) for label,index in label_index.items()}
    total_count = sum(prior.values())

    for label in prior:
        prior[label] = prior[label]/float(total_count)
    return prior
""" Compute likelihood based on training samples
Args:
    1.term_document_matrix (sparse matrix)
    2.label_index (grouped sample indices by class)
    3.smoothing (integer, additive Laplace smoothing parameter)
Returns:
    dictionary, with class as key, corresponding conditional probability P(feature|class) vector as value
"""
def get_likelihood(term_document_matrix, label_index, smoothing):
    likelihood = {}
    for label, index in label_index.items():
        likelihood[label] = term_document_matrix[index,:].sum(axis = 0) + 1
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label]/float(total_count)
    return likelihood

def get_posterior(term_document_matrix, prior, likelihood) :
	num_docs = term_document_matrix.shape[0]
	posteriors = []
	for i in range(num_docs) :
	# posterior is proportional to prior * likelihood
	#  = exp(log(prior * likelihood))
	#  = exp(log(prior) + log(likelihood))
		posterior = {key : np.log(prior_label)for key, prior_label in prior.items()}
		for label, likelihood_label in likelihood.items() :
			term_document_vector = term_document_matrix.getrow(i)
			counts = term_document_vector.data
			indices = term_document_vector.indices
			for count, index in zip(counts, indices) :
				posterior[label] += np.log(likelihood_label[index]) * count
		# exp(-1000) : exp(-999)will cause zero division error,
		# however it equates to exp(0) : exp(1)
		min_log_posterior = min(posterior.values())
		for label in posterior :
			try  :
				posterior[label] = np.exp(posterior[label] - min_log_posterior)
			except :
				# if one 's log value is excessively large, assign it infinity
				posterior[label] = float(' inf ')
			# normalize so that all sums up to 1
		sum_posterior = sum(posterior.values())
		for label in posterior:
			if posterior[label] == float(' inf '):
				posterior[label] = 1.0
			else :
			    posterior[label] /= sum_posterior
		posteriors.append(posterior.copy())
	return posteriors

#Collecting all the spam emails
file_path = "D:/personalWorkspace/ML_Practice/enron1/spam/"
for file in glob.glob(os.path.join(file_path,'*.txt')):
    with open(file, 'r', encoding = 'ISO-8859-1') as infile:
        emails.append(infile.read())
        labels.append(1)

#Collecting all the ham emails
file_path = "D:/personalWorkspace/ML_Practice/enron1/ham/"
for file in glob.glob(os.path.join(file_path,'*.txt')):
    with open(file, 'r', encoding = 'ISO-8859-1') as infile:
        emails.append(infile.read())
        labels.append(0)

cleaned_emails = clean_text(emails)
print(cleaned_emails[0])
cv = CountVectorizer(stop_words = "english", max_features = 500)
term_docs = cv.fit_transform(cleaned_emails)
label_index = get_label_index(labels)
prior = get_prior(label_index)
smooth = 1
likelihood = get_likelihood(term_docs,label_index,smooth)
emails_test = ['''Subject: flat screens hello , please call or contact regarding the other flat screens requested . trisha tlapek - eb 3132 b michael sergeev - eb 3132 a also the sun blocker that was taken away from eb 3131 a . trisha should two monitors also michael . thanks kevin moore''', '''Subject: having problems in bed ? we can help ! cialis allows men to enjoy a fully normal sex life without having to plan the sexual act . if we let things terrify us, life will not be worth living brevity is the soul of lingerie . suspicion always haunts the guilty mind .''']
print(emails_test)
cleaned_test = clean_text(emails_test)
term_docs_test = cv.transform(cleaned_test)
posterior = get_posterior(term_docs_test, prior, likelihood)
print(posterior)
