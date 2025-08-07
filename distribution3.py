import matplotlib.pyplot as plt

import os
import numpy as np
from scipy.stats import norm, gamma
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression

from pyfrechet_k.metric_spaces import MetricData
from pyfrechet_k.metric_spaces.sphere import Sphere, r2_to_angle
from pyfrechet_k.regression.frechet_regression import GlobalFrechet, LocalFrechet
from pyfrechet_k.metric_spaces import MetricData, Wasserstein1D
from pyfrechet.regression.kernels import NadarayaWatson, gaussian, epanechnikov

from adaboostval_4 import AdaBoost
from bikeplots import prepare_folder


np.random.seed(12) # adaboost improves
np.random.seed(15)
state = np.random.get_state()
print("random state ", state[1][0])

# set and clear figures folder
fig_folder = "fig_folder/distribution"
prepare_folder(fig_folder)

#np.random.normal()
mu, sigma = 0, 1
samp_g = norm.rvs(loc = mu, scale = sigma, size = 100)

plt.plot(samp_g)
#plt.show()
n_obs = 500

def make_data(x1, mu0=0, b=3, v1=0.25, sigma0=3, gamma0=0.5, v2=1):
    n_features = 200
    n_obs = len(x1)

    mean_mu_x = mu0 + b*x1
    std_mu_x = v1
    mu_x = norm.rvs(loc = np.array([mean_mu_x]*n_features).T, scale = np.array([std_mu_x]*n_features).T, size=(n_obs, n_features))

    shape_sigma_x = (sigma0 + gamma0*x1)**2/v2
    scale_sigma_x = v2/(sigma0 + gamma0*x1)
    sigma_x = gamma.rvs(a  = np.array([shape_sigma_x]*n_features).T, scale =  np.array([scale_sigma_x]*n_features).T,size=(n_obs, n_features))

    #samps = norm.rvs(loc = mu2, scale = sigma2, size = (n_obs, n_features))
    #samps = norm.rvs(loc = np.array([mu_x]*n_features).T, scale = np.array([sigma_x]*n_features).T, size = (n_obs, n_features))
    samps = norm.rvs(loc = mu_x, scale = sigma_x)

    #fig = plt.figure()
    #plt.hist(samps[0,:])
    #fig = plt.figure()
    #plt.hist(samps[-1,:])
    #plt.show()

    qs =  np.linspace(0,1,100)
    quants = np.quantile(samps, axis = 1, q=qs).T
    #plt.figure(); plt.plot(quants.T, alpha=0.5); plt.show()

    X1 = x1.reshape(-1,1)
    #X2 = x2.reshape(-1,1)
    Y = MetricData(M = Wasserstein1D(100), data = quants)
    #X1 += norm.rvs(loc = 0, scale = noise, size = X1.shape)
   #X2 += norm.rvs(loc = 0, scale = noise, size = X2.shape)

    X = X1
    return X, Y

def make_data2(mu0=0, b=3, v1=0.25, sigma0=3, gamma0=0.5, v2=1):
    x = np.random.uniform(low=-1, high= 1)
    
    mean_mu_x = mu0 + b*x
    std_mu_x = v1
    mu_x = norm.rvs(loc = mean_mu_x, scale = std_mu_x)

    shape_sigma_x = (sigma0 + gamma0*x)**2/v2
    scale_sigma_x = v2/(sigma0 + gamma0*x)
    sigma_x = gamma.rvs(a  = shape_sigma_x, scale =  scale_sigma_x)

    qs =  np.linspace(0 + np.finfo(np.float16).eps,1 -np.finfo(np.float16).eps,100) # machine epsilon

    y = mu_x + sigma_x*norm.ppf(qs, loc = 0,scale = 1)
    y2 = norm.ppf(qs, loc = mu_x, scale = sigma_x)
    #y = MetricData(M = Wasserstein1D(100), data = ys)
    return x, y

def make_data2_multiple(n_samp = 500):
    x_samp_norm = np.zeros(n_samp)
    y_samp_norm = np.zeros((n_samp,100))
    for i in range(n_samp):
        made_data = make_data2()
        x_samp_norm[i] =  made_data[0]
        y_samp_norm[i,:] =  made_data[1]

    X, Y = np.expand_dims(x_samp_norm, axis=1), MetricData(M = Wasserstein1D(100), data = y_samp_norm) 
    return X, Y

def make_data3(x=None, mu0=0, b=3, v1=0.25, sigma0=3, gamma0=0.5, v2=1):
    if x is None:
        x = np.random.uniform(low=-1, high= 1)
        #x = np.sin(x**2)
    n_quant_samples = 100
    
    mean_mu_x = mu0 + b*x
    std_mu_x = v1
    mu_x = norm.rvs(loc = mean_mu_x, scale = std_mu_x)

    shape_sigma_x = (sigma0 + gamma0*x)**2/v2
    scale_sigma_x = v2/(sigma0 + gamma0*x)
    sigma_x = gamma.rvs(a  = shape_sigma_x, scale =  scale_sigma_x)

    qs =   np.linspace(0 + np.finfo(np.float16).eps,1- np.finfo(np.float16).eps,100) # machine epsilon
    qss =  np.linspace(0 + np.finfo(np.float16).eps,1 -np.finfo(np.float16).eps ,100)
    #quants = np.quantile(samps, axis = 1, q=qs).T
    y_samples = norm.rvs(size=n_quant_samples, loc = mu_x, scale = sigma_x)
    y1 = np.quantile(y_samples, q=qss)
    y = mu_x + sigma_x*norm.ppf(qs, loc = 0,scale = 1)
    #y2 = norm.ppf(qs, loc = mu_x, scale = sigma_x)
    #y = MetricData(M = Wasserstein1D(100), data = ys)
    return x, y, np.hstack((mu_x, sigma_x))

def make_data3_multiple(x=500):
    # x: array -> ignore n_samp, and calculate for x (all)
    # x: number -> sample x, and calculate for x (all)

    if isinstance(x, type(1)):
        # x is an integer
        n_samp = x
        x_samp_norm = np.zeros(n_samp)
        y_samp_norm = np.zeros((n_samp,100))
        mu_sigma = np.zeros((n_samp, 2))
        for i in range(n_samp):
            made_data = make_data3()
            x_samp_norm[i] =  made_data[0]
            y_samp_norm[i,:] =  made_data[1]
            mu_sigma[i,:] = made_data[2]
    else:
        # x is array like
        made_data = make_data3(x)
        x_samp_norm =  made_data[0]
        y_samp_norm =  made_data[1]
        mu_sigma = made_data[2]    

    X = x_samp_norm
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=1)
        
    Y = MetricData(M = Wasserstein1D(100), data = y_samp_norm) 
    return X, Y, mu_sigma

#def makedata2det(x, n_samp=500):

    

n_samp = 500
X, Y, mu_sigma = make_data3_multiple(n_samp)

#x1_test100 = np.random.randint(600, size = 100)/100
#x_test100 = np.random.uniform(low=-1, high=1, size = n_obs)
#x2_test100 = np.random.randint(200, size = 100)/100
#x_test = x_test100.reshape(-1,1)
#x_test = np.hstack((x_test, x_test**2))
X_test, Y_test, mu_sigma_test = make_data3_multiple(n_samp)
#print(X_test.shape)

#Fitting GlobalFrechet
gf_sim = GlobalFrechet().fit(X,Y)
pred_train = gf_sim.predict(X) # predict with fitted model
print("GlobalFrechet training M.d loss ", np.mean(Y.M.d(Y.data, pred_train.data)))
#Testing fit GlobalFrechet
pred_test = gf_sim.predict(X_test)
print("GlobalFrechet test M.d loss  ", np.mean(Y_test.M.d(Y_test.data, pred_test.data)))


#Fitting GlobalFrechet with AdaBoost

Wasserstein_dim = 100
max_learners = 20
ada_train_data = np.hstack((X,Y))
ada_sim = mywBooster = AdaBoost(
            data = ada_train_data, 
            learner = GlobalFrechet, 
            omega = Wasserstein1D(Wasserstein_dim), 
            t = max_learners,
            dimY = Wasserstein_dim)
 
#ada_sim = GlobalFrechet().fit(X,Y)
ada_sim.fit()
ada_test_data = np.hstack((X_test,Y_test))

pred_train = ada_sim.predict(ada_train_data, "mean_fm")[0] # predict with fitted model
print("mean_fm training M.d loss ", np.mean(Y.M.d(Y.data, pred_train.data)))
pred_test = ada_sim.predict(ada_test_data, "mean_fm")[0]
print("mean_fm test M.d loss  ", np.mean(Y_test.M.d(Y_test.data, pred_test.data)))

# Different learners
pred_test_ada = ada_sim.predict(ada_test_data, "mean_fm")[0]
pred_test_all = ada_sim.predict(ada_test_data, "mean_fm")[1]

#pred_test_all.shape[:,0,:]
#all_mu = pred_test_all[:,0,50]

if 0: # demonstration. calculate mu and sigma from quantile plot
    pred_ = norm.ppf((np.linspace(0+1e-6, 1-1e-6, 100)), loc=3, scale=2)
    pred_ = 3 + 2*norm.ppf((np.linspace(0+1e-6, 1-1e-6, 100)))
    mu_ = np.mean(pred_)
    sigma_ = np.mean((pred_-mu_)/norm.ppf(np.linspace(0+1e-6, 1-1e-6, 100)))

def musig_from_quant(quant, eps=1e-6):
    mu =  (quant[49] + quant[50])/2 #np.mean(quant) 
    sigma = np.mean((quant-mu)/norm.ppf(np.linspace(0+eps, 1-eps, 100)))
    return mu, sigma

i = 12
eps=1e-6
mu_data = mu_sigma[i,0] 
sigma_data = mu_sigma[i,1]
print("data", "mu_data,  sigma_data ", mu_data, sigma_data)

quant = pred_test_ada.data[i,:]
#print("ex error", ada_sim.omega.d(quant,norm.ppf(np.linspace(0+eps, 1-eps, 100), loc=mu_data, scale=sigma_data)))
mu_quant, sigma_quant = musig_from_quant(quant)
print("adaB", "mu_quant, sigma_quant", mu_quant, sigma_quant)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Adaboost")
ax1.scatter(quant, norm.ppf(np.linspace(0+eps, 1-eps, 100), loc=mu_data, scale=sigma_data), label="pred vs data musig")
ax1.legend()

linspace = np.linspace(mu_quant-3*sigma_quant, mu_quant+3*sigma_quant, 100)
ax2.plot(linspace, norm.pdf(linspace, loc=mu_quant, scale=sigma_quant), label="fitted")
linspace = np.linspace(mu_data -3*sigma_data,  mu_data+3 *sigma_data,  100)
ax2.plot(linspace, norm.pdf(linspace, loc=mu_data,  scale=sigma_data), "--", label="dataset")
ax2.legend()

fig.tight_layout()
fig.savefig(os.path.join(fig_folder, str(i)+" Adaboost.pdf"))
fig.show()

for t in range(ada_sim.t):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (7,4))

    quant = pred_test_all[t,i,:]
    mu_quant, sigma_quant = musig_from_quant(quant)
    print(t+1, "mu_quant, sigma_quant", mu_quant, sigma_quant)
#    ax1.scatter(quant, norm.ppf(np.linspace(0+eps, 1-eps, 100), loc=mu_quant, scale=sigma_quant), label="fitted musig")
    ax1.scatter(quant, norm.ppf(np.linspace(0+eps, 1-eps, 100), loc=mu_data, scale=sigma_data)) #, label="Predicted vs theoretical quantiles")
    ax1.title.set_text("Predicted vs theoretical quantiles")
    #ax1.legend(loc="upper left")

    linspace = np.linspace(mu_quant-3*sigma_quant, mu_quant+3*sigma_quant, 100)
    ax2.plot(linspace, norm.pdf(linspace, loc=mu_quant, scale=sigma_quant), label="Fitted pdf")
    linspace = np.linspace(mu_data -3*sigma_data,  mu_data+3 *sigma_data,  100)
    ax2.plot(linspace, norm.pdf(linspace, loc=mu_data,  scale=sigma_data), "--", label=" Pdf of target distribution")
    ax2.legend(loc="upper left")

    fig.suptitle(f"Learner  {t}")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_folder, str(i)+ " learner "+str(t)+".pdf"))
    fig.show()

#plt.show()

pred_train = ada_sim.predict(ada_train_data, "mean_weight")[0] # predict with fitted model
print("mean_weight training M.d loss ", np.mean(Y.M.d(Y.data, pred_train.data)))
pred_test = ada_sim.predict(ada_test_data, "mean_weight")[0]
print("mean_weight test M.d loss  ", np.mean(Y_test.M.d(Y_test.data, pred_test.data)))



#Fitting LinearRegression
lr_sim = LinearRegression().fit(X,mu_sigma)
lr_pred_train = lr_sim.predict(X) # predict with fitted model
# lr_loss = 0 #np.mean(Y.M.d(Y.data, lr_pred_train.data))
# print("training loss  ", lr_loss)
#Testing fit GlobalFrechet
lr_pred_test = lr_sim.predict(X_test)
# lr_loss_test = 0 #np.mean(Y_test.M.d(Y_test.data, lr_pred_test.data))
# print("test loss  ", lr_loss_test)



x_ = np.linspace(0, 1, 100)
x_plot, y_plot, mu_sigma_plot = make_data3_multiple(1)
gf_plot = gf_sim.predict(x_plot)
fig = plt.figure()
plt.plot(x_, y_plot.data[0], label="y")
plt.plot(x_, gf_plot, label="gf")
plt.legend()
plt.title("y_plot vs gf_plot, x="+str(x_plot))
fig.savefig(os.path.join(fig_folder, "y_plot vs gf_plot.pdf"))
fig.show()


### ISE
def m_regression_function(x, grid, mu0=0, b=3, sigma0=3, gamma0=0.5):
    #x = np.linspace(0,1,n)
    grid_ = grid
    grid_[0]  += np.finfo(np.float16).eps
    grid_[-1] -= np.finfo(np.float16).eps
    return mu0 + b*x + (sigma0 + gamma0*x)*norm.ppf(grid_, loc = 0,scale = 1)


def predictmusig(x, model):
    x= np.asarray(x).reshape(-1,1)
    musig = model.predict(x)
    grid_ = np.linspace(0,1,100)
    grid_[0]  += np.finfo(np.float16).eps
    grid_[-1] -= np.finfo(np.float16).eps
    y_pred =  MetricData(M = Wasserstein1D(100), data =musig[:,0] + musig[:,1] *norm.ppf(grid_, loc = 0,scale = 1))
    y_test = MetricData(M = Wasserstein1D(100), data = m_regression_function(x, y_pred.M.GRID))
    
    return y_test.M.d(y_test.data,y_pred.data)**2

def predictfun(x, model):
    x= np.asarray(x).reshape(-1,1)
    y_pred = MetricData(M = Wasserstein1D(100), data = model.predict(x))
    y_test = MetricData(M = Wasserstein1D(100), data = m_regression_function(x, y_pred.M.GRID))
    
    return y_test.M.d(y_test.data,y_pred.data)**2

def predictadafun(x, model, arg):
    x= np.asarray(x).reshape(-1,1)
    y_ = np.zeros((x.shape[0], model.dimY))
    x_ = np.hstack((x,y_))
    y_pred = MetricData(M = Wasserstein1D(100), data = model.predict(x_,arg)[0])
    y_test = MetricData(M = Wasserstein1D(100), data = m_regression_function(x, y_pred.M.GRID))
    
    return y_test.M.d(y_test.data,y_pred.data)**2

ise = quad(func=lambda x: predictfun(x, gf_sim), a=-1, b=1)
print("ise", ise) 

ise_musig = quad(func=lambda x: predictmusig(x, lr_sim), a=-1, b=1)
print("ise musig", ise_musig) 

Y_test.M.d(Y_test.data, pred_test.data)**2

fig = plt.figure()
plt.plot(np.linspace(0,1,len(X)), np.sort(X[:,0]), "*", label="train")
plt.plot(np.linspace(0,1,len(X_test)), np.sort(X_test[:,0]), "o", label="Test")
plt.legend()
plt.title("Train vs test data\n(sorted)")
fig.savefig(os.path.join(fig_folder, "Train vs test data (sorted).pdf"))
fig.show()


## Train
#for i in np.random.choice(range(len(X)), size=2):
#    fig = plt.figure()
#    plt.plot(pred_train.data[i,:], label="pred " + str(X[i]))
#    plt.plot(Y.data[i,:], label="Y " + str(X[i]))
#    plt.legend()
#    plt.title("Train")
#    fig.show()

## Test
#for i in range(2):
#    fig = plt.figure()
#    plt.plot(pred_test.data[i,:], label="pred " + str(X_test[i]))
#    plt.plot(Y_test.data[i,:], label="Y_test " + str(X_test[i]))
#    plt.legend()
#    plt.title("Test")
#    fig.show()


'''
n_iter = 200
n_samples = [50, 100, 200]
ises = np.empty((len(n_samples),n_iter))
ises[:] = np.nan

for i, n_samp in enumerate(n_samples):
    print("started iteration", i, "- n_samp =", n_samp)
    for j in range(n_iter):
        print("...", j, "of", n_iter, end="\r")
        X, Y, mu_sigma = make_data3_multiple(n_samp)
        gf_sim = GlobalFrechet().fit(X,Y)
        ise = quad(func=lambda x: predictfun(x, gf_sim), a=-1, b=1)
        ises[i,j] = ise[0]
    print() 

fig = plt.figure()
plt.boxplot(ises.T)
plt.title('ises')
fig.show()
'''



n_iter = 200
n_samples = [50, 100, 200]
ises = np.empty((len(n_samples),n_iter))
ises_musig = np.empty((len(n_samples),n_iter))
ises_adafm = np.empty((len(n_samples),n_iter))
ises_adamw = np.empty((len(n_samples),n_iter))

ises[:] = np.nan
ises_musig[:] = np.nan
ises_adafm[:] = np.nan
ises_adamw[:] = np.nan


for i, n_samp in enumerate(n_samples):
    print("started iteration", i, "- n_samp =", n_samp)
    for j in range(n_iter):
        print("...", j, "of", n_iter, end="\r")
        X, Y, mu_sigma = make_data3_multiple(n_samp)
        gf_sim = GlobalFrechet().fit(X,Y)
        lr_sim = LinearRegression().fit(X,mu_sigma)
        
        ada_sim = mywBooster = AdaBoost(
            data = np.hstack((X,Y)), 
            learner = GlobalFrechet, 
            omega = Wasserstein1D(Wasserstein_dim), 
            t = max_learners,
            dimY = Wasserstein_dim)
        
        ada_sim.fit()

        ise_adafm = quad(func=lambda x: predictadafun(x, ada_sim, "mean_fm"), a=-1, b=1)
        ise_adamw = quad(func=lambda x: predictadafun(x, ada_sim, "mean_weight"), a=-1, b=1)

        ise_musig = quad(func=lambda x: predictmusig(x, lr_sim), a=-1, b=1)

        ise = quad(func=lambda x: predictfun(x, gf_sim), a=-1, b=1)
        ises[i,j] = ise[0]
        ises_musig[i,j] = ise_musig[0]
        ises_adafm[i,j] = ise_adafm[0]
        ises_adamw[i,j] = ise_adamw[0]
    print() 
costum_labels = ["50 sim", "100 sim", "200 sim"]


fig, ax = plt.subplots()
ax.boxplot(ises.T)
ax.set_xticks(np.arange(1, len(costum_labels)+1))
ax.set_xticklabels(costum_labels)
plt.title('ISE Global FR')
fig.savefig(os.path.join(fig_folder, "ISE Global FR.pdf"))
fig.show()

fig, ax = plt.subplots()
plt.boxplot(ises_musig.T)
ax.set_xticks(np.arange(1, len(costum_labels)+1))
ax.set_xticklabels(costum_labels)
plt.title('ISE oracle LR')
fig.savefig(os.path.join(fig_folder, "ISE oracle LR.pdf"))
fig.show()

fig, ax = plt.subplots()
plt.boxplot(ises_adafm.T)
ax.set_xticks(np.arange(1, len(costum_labels)+1))
ax.set_xticklabels(costum_labels)
plt.title('ISE AdaBoost FM')
fig.savefig(os.path.join(fig_folder, "ISE AdaBoost FM.pdf"))
fig.show()

fig, ax = plt.subplots()
plt.boxplot(ises_adamw.T)
ax.set_xticks(np.arange(1, len(costum_labels)+1))
ax.set_xticklabels(costum_labels)
plt.title('ISE AdaBoost summed weights')
fig.savefig(os.path.join(fig_folder, "ISE AdaBoost summed weights.pdf"))
fig.show()

plt.show()
print("Fin")



