#Athanasios Kousathanas
#Simple linear Bayesian regression with Edward/tensorflow probabilistic framework

import numpy as np
import matplotlib.pyplot as plt
import edward as ed
import tensorflow as tf

from edward.models import Normal

#simulate data
#individuals N and Markers M
M=100
N=500

#intercept true_alpha
true_alpha=5

#standard deviations for covariates (true_sigma_b) and for noise (true_sigma_e)
true_sigma_b=2
true_sigma_e=1

#covariates true_beta of dimension M
true_beta=np.random.normal(1.0, true_sigma_b, size=M)
#noise epsilon of dimension N
epsilon=np.random.normal(0.0, true_sigma_e, size=N)

#genotype matrix X_train
X_train = np.random.normal(0.0, 1.0, size=(N, M))
#Phenotypes Y_train
Y_train = np.dot(X_train, true_beta) + true_alpha + epsilon

#Define Bayesian regression model
X = tf.placeholder(tf.float32, [N, M])

sigma_e= tf.Variable(1.0)
sigma_b = tf.Variable(1.0)

beta = Normal(loc=tf.zeros(M), scale=sigma_b*tf.ones(M))
alpha = Normal(loc=tf.zeros(1), scale=tf.ones(1))

Y = Normal(loc=ed.dot(X, beta ) + alpha, scale=sigma_e*tf.ones(N))

qbeta = Normal(loc=tf.get_variable("qbeta/mean", [M]),
            scale=tf.nn.softplus(tf.get_variable("qbeta/scale", [M])))
qalpha = Normal(loc=tf.get_variable("qalpha/mean", [1]),
            scale=tf.nn.softplus(tf.get_variable("qalpha/scale", [1])))

#run black-box variational inference
inference = ed.KLqp({beta: qbeta, alpha: qalpha}, data={X: X_train, Y: Y_train})
inference.run(n_samples=10, n_iter=10000)

sess = ed.get_session()
#obtain posterior mean for beta and alpha
Ebeta=sess.run(qbeta.mean())
Ealpha=sess.run(qalpha.mean())

plt.scatter(Ebeta,true_beta)
plt.title('Estimated versus true betas')
plt.xlabel('Estimated')
plt.ylabel('True')

Y_est=np.dot(X_train,Ebeta)+Ealpha
plt.scatter(Y_est,Y_train)
plt.title('Estimated versus true Y')
plt.xlabel('Estimated')
plt.ylabel('True')

#Rest of estimated parameters
print("E(alpha)=",Ealpha,",","True=",true_alpha)
print("E(sigma_e)=",sigma_e.eval(),",","True=",true_sigma_e)
print("E(sigma_b)=",sigma_b.eval(),",","True=",true_sigma_b)
