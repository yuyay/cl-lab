# encoding=UTF-8
intro = """
This introduction has not written yet.
"""
# module import
from optparse import OptionParser
import numpy as np
import newsfeatures
import feedparser
import sys
from scipy.special import gammaln

# option settings
# detail: http://pythonjp.sourceforge.jp/dev/library/optparse.html
usage = "usage: %prog [options] \n"+intro; # 
parser = OptionParser(usage=usage);
"""
parser.add_option("-f", "--file", dest="filename", help="write report to FILE", metavar="FILE");
parser.add_option("-q", "--quiet", action="store_false", dest="verbose", default=True, help="don't print status messages to stdout");
"""
option,args = parser.parse_args();

class LDA(object):
	def __init__(self, K, docs, alpha=0.5, beta=0.5):
		self.K = K # number of topics
		self.docs = docs 	# i.e. docs = [[the, boy, has, a, cat,...],...]
		self.M = len(docs) # ドキュメントの数
		self.n_m = np.zeros(self.M, dtype='int32') # ドキュメントm内の単語の数
		self.term_id = {} # {term t in vocabulary: id}
		self.z_mn = [0 for i in range(self.M)] # ドキュメントm内の単語nのトピック (z_mn[m,n] = k)
		self.n_mk = np.zeros((self.M, self.K), dtype='int32') # ドキュメントm内でトピックkの単語が表れた回数
		"""make term-id dictionary"""
		self.V = 0 # number of terms (number of vocabulary)
		for m in range(self.M):
			for word in docs[m]:
				if self.term_id.has_key(word): pass
				else:
					self.term_id[word] = self.V
					self.V+=1
		self.n_kt = np.zeros((self.K, self.V), dtype="int32") # トピックkの語彙tの数
		self.n_k = np.zeros(self.K, dtype="int32") # トピックkの語彙数
		if type(alpha) == type(list()): self.alpha = np.array(alpha) # hyperparameter of topic-word distribution
		else: self.alpha = alpha*np.ones(self.K)
		if type(beta) == type(list()): self.beta = np.array(beta) # hyperparameter of 
		else: self.beta = beta*np.ones(self.V)

		# initialize variable
		for m in range(self.M):
			self.n_m[m] = len(docs[m]);
			self.z_mn[m] = [0 for i in range(self.n_m[m])]
			for n,word in enumerate(docs[m]):
				k = np.random.randint(0, K) # initial topic is uniformly random
				self.z_mn[m][n] = k
				self.n_mk[m, k] += 1
				self.n_kt[k, self.term_id[word]] += 1 
				self.n_k[k] += 1

	# inference using Gibbs sampling
	def inference(self):
		sum_beta = self.beta.sum()
		sum_alpha = self.alpha.sum()
		for m in range(self.M):
			term2_lower = self.n_mk[m, :].sum() + sum_alpha - 1 
			term2_upper = self.n_mk[m, :] + self.alpha
			for n,word in enumerate(self.docs[m]):
				# discount
				k = self.z_mn[m][n]
				self.n_mk[m, k] -= 1; self.n_m[m] -= 1; self.n_kt[k, self.term_id[word]] -= 1; self.n_k[k] -= 1; 
				# sampling new topic k with full condition
				p_z = term2_upper*(self.n_kt[:, self.term_id[word]].T + self.beta[self.term_id[word]])
				p_z = p_z / (term2_lower * (self.n_kt.sum(axis=1) + sum_beta) )
				
				"""
				p_z2 = [0.0 for k in range(self.K)]
				for z in range(self.K):
					term2_upper = self.n_mk[m, z] + self.alpha[z]
					term1_upper = self.n_kt[z, self.term_id[word]] + self.beta[self.term_id[word]]
					term1_lower = sum_beta + self.n_kt[z].sum()
					p_z2[z] = (term1_upper/term1_lower)*(term2_upper/term2_lower)
					print p_z[z], p_z2[z]
				"""

				k_new = np.random.multinomial(1, p_z/p_z.sum()).argmax();	
				self.z_mn[m][n] = k_new

				# increment with new topik k_new
				self.n_mk[m, k_new] += 1; self.n_m[m] += 1; self.n_kt[k_new, self.term_id[word]] += 1; self.n_k[k_new] += 1; 

	# parameter calculation
	def calc_parameter(self):
		phi_kt = np.zeros((self.K, self.V))
		theta_mk = np.zeros((self.M, self.K))
		for k in range(self.K):
			summation = self.n_kt[k, :].sum() + self.beta.sum()
			phi_kt[k, :] = (self.n_kt[k, :]+self.beta)/summation
		for m in range(self.M):
			summation = self.n_mk[m, :].sum() + self.alpha.sum()
			theta_mk[m, :] = (self.n_mk[m, :] + self.alpha) / summation
	
		return phi_kt, theta_mk;

	def calc_likelihood(self):
		loglik = 0;
		alpha = self.alpha[0];
		beta = self.beta[0];
		M = len(self.docs);
		K = self.K;
		T = self.V;
		loglik += M*(gammaln(alpha*K) - K*(gammaln(alpha)))
		loglik += self.n_mk.sum()
		loglik -= sum([gammaln(sum([self.n_mk[m, k] for k in range(K)]) + alpha*K) for m in range(M)])
		loglik += K*(gammaln(beta*T) - gammaln(beta))
		loglik += sum([sum([gammaln(self.n_kt[k, t] + beta) for k in range(K)]) for t in range(T)])
		loglik -= sum([gammaln(sum([self.n_kt[k, t] for t in range(T)]) + beta*T)])
		return loglik;

	def print_result(self, phi_kt, theta_mk):
		id_term = dict([(id,term) for term,id in self.term_id.items()]); 
		#print phi_k_t
		for k in range(self.K):
			p_term = [(phi_kt[k, t], t) for t in range(self.V)]
			print "Topic %d contains:"%(k+1)
			p_term.sort(reverse=True)
			for i in range(5):
				print p_term[i][0], id_term[p_term[i][1]], self.n_kt[k, p_term[i][1]]

def get_sample_article(output="article.txt"):
	articletitles = [];
	articlewords = [];
	for feed in newsfeatures.feedlist:
		f = feedparser.parse(feed)
		# 全ての記事をループする                                                                                                                                                                        
		for e in f.entries:
			# 同一の記事は無視                                                   
			if e.title in articletitles: continue;
			# 単語を抽出する
			txt = e.title.encode('utf8')+newsfeatures.stripHTML(e.description.encode('utf8'));
			articlewords.append(newsfeatures.separatewords(txt));
			articletitles.append(e.title);
	f = open(output, "w");
	f.writelines([" ".join(articlewords[i])+"\n" for i in range(len(articlewords))])
	#return articlewords, articletitles;

# main
def main():
	fn = args[0]; # argv[1]: document file
	K = int(args[1]); # argv[2]: number of topic
	f = open(fn, "r");
	docs = [];
	for line in f.readlines():
		docs.append(line.strip().split(" "));
	f.close();

	lda = LDA(K, docs);
	for i in range(100):
		print "Iteration %d"%(i+1);
		lda.inference();
		phi_kt, theta_mk = lda.calc_parameter();
		print "LogLikelihood: %e"%(lda.calc_likelihood())
	
		lda.print_result(phi_kt, theta_mk)
	#print lda.term_id,lda.n_k_t;
	
if __name__ == "__main__":
	main();

