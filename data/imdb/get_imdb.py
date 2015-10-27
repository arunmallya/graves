# adapted from http://deeplearning.net/tutorial/code/imdb.py

import cPickle
import os

def get_dataset_file(location, origin):
	if not os.path.isfile(location): 
	    import urllib
	    print 'Downloading data from %s' % origin
	    urllib.urlretrieve(origin, location)


# load data
filename = 'imdb.pkl'
get_dataset_file(filename, 
	'http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl')

fin = open(filename, 'rb')
train_set = cPickle.load(fin)
test_set  = cPickle.load(fin)
fin.close()

# write out to text file
train_data  = train_set[0]
train_label = train_set[1]
test_data  = test_set[0]
test_label = test_set[1]
assert len(train_data) == len(train_label)
assert len(test_data) == len(test_label)

print 'Num train samples =', len(train_data)
print 'Num test samples =', len(test_data)

print 'Writing train data to train.txt'
with open('train.txt', 'w') as fout:
	for i in range(len(train_data)):
		fout.write('%d,'%train_label[i])
		for ind, item in enumerate(train_data[i]):
			if ind == len(train_data[i])-1:
				if i == len(train_data)-1:
					fout.write('%d'%item)
				else:
					fout.write('%d\n'%item)
			else:
				fout.write('%d,'%item)

print 'Writing test data to test.txt'
with open('test.txt', 'w') as fout:
	for i in range(len(test_data)):
		fout.write('%d,'%test_label[i])
		for ind, item in enumerate(test_data[i]):
			if ind == len(test_data[i])-1:
				if i == len(train_data)-1:
					fout.write('%d'%item)
				else:
					fout.write('%d\n'%item)
			else:
				fout.write('%d,'%item)
