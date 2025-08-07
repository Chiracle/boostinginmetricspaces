import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pyfrechet_k.metric_spaces import MetricData, Wasserstein1D
from pyfrechet_k.metrics import mse
from pyfrechet_k.regression.frechet_regression import GlobalFrechet

from adaboostval_4 import AdaBoost 

import os
import shutil

def prepare_folder(path):
    if os.path.exists(path):
        for entry in os.listdir(path):
            full = os.path.join(path, entry)
            if os.path.isfile(full) or os.path.islink(full):
                os.remove(full)
            elif os.path.isdir(full):
                shutil.rmtree(full)
    # (re)create folder
    if not os.path.exists(path):
        os.makedirs(path)

Wasserstein_dim = 100
Wasserstein_grid = np.linspace(0, 1, Wasserstein_dim)
# Machine epsilon begin
eps_div = 0 #np.finfo(np.float16).eps*1000

# set and clear figures folder
fig_folder = "fig_folder/bikeplots"
prepare_folder(fig_folder)

random_state =  None # seed
shuffle = True


def hist_from_quant(quants, ys, sample_multiplier=1e6, eps_div = 1e-6):
    midpoints = (quants[1:] + quants[:-1]) / 2
    quants_diff = np.diff(quants)
    quants_diff = np.where(quants_diff == 0, eps_div, quants_diff) 
    pdf = np.diff(ys) / quants_diff

    sim_samples = []
    for i in range(len(midpoints)):
        n_repeats = int((pdf[i] * np.diff(quants)[i] * sample_multiplier))
        sim_samples.extend(np.repeat(midpoints[i], n_repeats))

    return np.asarray(sim_samples)

def make_base_plots(prefix, pred_test, y_test, pred_train, y_train, quantiles_list, ax_titles):
  ## predictions for representative quantiles
  # code to find error predictions
  sq_err_test = y_test.M.d(y_test, pred_test)**2
  if pred_train is not None:
    sq_err_train = y_train.M.d(y_train, pred_train)**2

  se_low_quartile = np.quantile(sq_err_test, q=quantiles_list[0])
  se_median = np.quantile(sq_err_test, q=quantiles_list[1])
  se_up_quartile = np.quantile(sq_err_test, q=quantiles_list[2])

  idx_se_low_q = np.argmin(np.abs(sq_err_test - se_low_quartile))
  idx_se_median_q = np.argmin(np.abs(sq_err_test - se_median))
  idx_se_up_q = np.argmin(np.abs(sq_err_test - se_up_quartile))
  idx_list = [idx_se_low_q, idx_se_median_q, idx_se_up_q]

  n_subplots = len(idx_list)

  fig, axes = plt.subplots(1, n_subplots, sharey=True)
#  fig.suptitle("eCDF")
  for i in range(n_subplots):
    ax = axes[i]
    idx = idx_list[i]
    ax.set_title(ax_titles[i] + "\n" + x_test.index[idx])

    # eCDF pred
    x_lin = pred_test.data[idx]
    x_lin = np.insert(x_lin, -1, x_lin[-1])
    ys = np.arange(1,Wasserstein_dim+1) / (len(x_lin))#[1:-1]
    ax.hlines(y = ys, xmin=x_lin[:-1], xmax=x_lin[1:],
            color='grey', zorder=1, label="Pred CDF")
    ax.scatter(x_lin[:-1], ys, color='grey', s=3, zorder=2)
    # eCDF test
    x_lin = y_test[idx]
    x_lin = np.insert(x_lin, -1, x_lin[-1])
    ys = np.arange(1,Wasserstein_dim+1) / (len(x_lin))#[1:-1]
    ax.hlines(y = ys, xmin=x_lin[:-1], xmax=x_lin[1:],
            color='lightgrey', zorder=1, label="Test eCDF")
    ax.scatter(x_lin[:-1], ys, color='lightgrey', s=3, zorder=2)

  handles, _ = axes[0].get_legend_handles_labels()
  labels = ["Pred CDF", "Test eCDF"]
  fig.legend(handles, labels, loc="lower center", ncol=len(labels))
  fig.tight_layout(rect=[0,0.1,1,1])
  fig.savefig(os.path.join(fig_folder, prefix + "_CDF.pdf"))
  fig.show()




  fig, axes = plt.subplots(n_subplots, 1, sharex=False, sharey=False)
  fig.suptitle("Histogram")
  for i in range(n_subplots):
    ax = axes[i]
#    ax.set_title(ax_titles[i] + "\n" + x_test.index[idx])
    idx = idx_list[i]

    ## Data prep
    date = x_test.index[idx]
    count = hours[hours['dteday'] == date]["cnt"]
    quant = pred_test.data[idx]
    ys = Wasserstein_grid

    ## Params
    n_bins = 15

    ## Histogram Calculations
    sim_samples = hist_from_quant(quant, ys)
    max_ = np.max((count.max(), sim_samples.max()))
    bins = np.linspace(0, max_, n_bins)

    ## Setup figure
#    fig, ax = plt.subplots()
#    fig.suptitle("Test vs Prediction")
    ax.hist(count,       bins=bins, density=True, alpha=0.7, color="silver",  label="Test")
    ax.hist(sim_samples, bins=bins, density=True, alpha=0.5, color="crimson", label="Prediction")
#    ax.legend()  

  '''
    ## test data, histogram densities
    date = x_test.index[idx]
    date_cnt = hours[hours['dteday'] == date]["cnt"]
    ax.hist(date_cnt, bins= "auto", density=True, alpha=1, color="silver", label="Test")

    ## pred, emperical densities
    ys = Wasserstein_grid
    x_lin = pred_test.data[idx]
    #x = np.where(x_lin == 0, np.finfo(np.float16).eps, x_lin)
    dx = np.diff(x_lin)
    dx = np.where(dx == 0, eps_div, dx)
    pdf_ = np.diff(ys) / dx
    midpoints = (x_lin[1:] + x_lin[:-1]) / 2
    ax.bar(midpoints, pdf_, width=0.1, align="center", alpha=0.8, color="black", edgecolor="black", label="Prediction")

  '''
  handles, _ = axes[0].get_legend_handles_labels()
  labels = ["Test data", "Predictions"]
  fig.legend(handles, labels, loc="lower center", ncol=len(labels))
  fig.tight_layout(rect=[0,0.1,1,1])
  fig.savefig(os.path.join(fig_folder, prefix + "_PDF.pdf"))
  fig.show()
  

  fig, axes = plt.subplots(1, n_subplots, sharey=True)
  fig.suptitle("Quantiles (test data)")
  for i in range(n_subplots):
    ax = axes[i]
    ax.set_title(ax_titles[i] + "\n" + x_test.index[idx])
    idx = idx_list[i]

    ax.plot(pred_test.data[idx], y_test[idx],  "o", mfc='none' ,color="darkgreen")
    #ax.scatter(pred_test.data[i], y_test[i])
    xs = np.linspace(0, np.max(y_test.data[idx,:]), 100)
    ys = xs
    ax.plot(xs, ys, "-", color= "lightgreen")

  fig.tight_layout()
  fig.savefig(os.path.join(fig_folder, prefix + "_QQ.pdf"))
  fig.show()

  return idx_list


def plot_QQ(pred, y, idx, label):
    figs = []
    for day in idx:
        fig, [ax_left, ax_right] = plt.subplots(ncols=2)
        title = f"{label}: D: {Wasserstein1D(Wasserstein_dim).d(y.data[day,:], pred.data[day,:]):0.4f}, day: {day}"
        fig.suptitle(title)

        #Quantile plot
#        fig = plt.figure()
        ax_left.plot(y.data[day,:], label = "y")
        ax_left.plot(pred.data[day,:], label = "prediction")
#        ax_left.set_title(label + str(Wasserstein1D(Wasserstein_dim).d(y.data[day,:], pred.data[day,:]))+ " - "+ "Day " +str(day))
        ax_left.legend()
#        figs.append(fig)

        #Q-Q plot
#        fig = plt.figure()
        ax_right.plot(y.data[day,:], pred.data[day,:], "o", mfc='none' ,color="darkgreen", label = "Q-Q" )
        #xs = np.linspace(0, np.max(np.array(np.max(y.data[day,:]),np.max(pred.data[day,:]))), 100)
        xs = np.linspace(0, np.max(y.data[day,:]), 100)

        ys = xs
        ax_right.plot(xs, ys, "-", color= "lightgreen")
        ax_right.legend()
        ax_right.set_xlabel("Empirical quantiles")
        ax_right.set_ylabel("Predicted quantiles")
#        ax_right.set_title(label +str(Wasserstein1D(Wasserstein_dim).d(y.data[day,:], pred.data[day,:]))+ " - "+ "Day " +str(day))

        fig.tight_layout()
        figs.append(fig)
    return figs


if __name__ == "__main__":


  #### based on code by MB
  #hours = pandas.read_csv('./bike+sharing+dataset/hour.csv')
  hours = pandas.read_csv('./hoursimp.csv')
  hours["quant"] = hours["cnt"]
  hours["cntday"] = hours["cnt"]
  hours["elaps"] = hours["yr"]*12 + hours['mnth']
  #  hours["weathersit"] = hours["weathersit"].apply(int)  # TODO: consider this
  hours["bw"] = hours["weathersit"]
  hours["rbw"] = hours["weathersit"]

  days = hours.groupby(by=['dteday'])
  days = days.aggregate({
      'season': 'min',
      'dteday': 'first',
      'yr': 'min', 
      'elaps': 'min',                
      'holiday': 'min',
      'weekday': 'min', 
      'workingday': 'min',
      'weathersit': 'max', # what is this?
      'temp': 'mean',
  #     'atemp': 'avg', # what is this?
      'hum': 'mean',
      'windspeed': 'mean',
      'bw': lambda x: np.max(np.where(x == 2, 1,0)),
      'rbw': lambda x: np.max(np.where(x == 3, 1,0)),
      'cnt': lambda cnt: [i for i in cnt],
      'cntday': lambda cnt: np.sum(cnt),
      'quant': lambda x: np.quantile(x, Wasserstein_grid).tolist()
      })

  hours["bw"] = hours["bw"].apply(lambda x: np.max(np.where(x == 2, 1,0)))
  hours["rbw"] = hours["rbw"].apply(lambda x: np.max(np.where(x == 3, 1,0)))


  var_list = ['yr','season','elaps','holiday','weekday','workingday','temp','hum','windspeed']
  #var_list = ['season','elaps','holiday','weekday','workingday','weathersit','temp','hum','windspeed']
  var_list = ['yr','workingday','temp','windspeed', 'bw', 'rbw']

  # var_list = ['season','holiday','weekday','workingday','weathersit']

  # Select only second year
  #hours = hours.loc[hours["yr"] == 1]

  dates = pandas.unique(hours["dteday"])
  dates_train, dates_test = train_test_split(
      dates, test_size=0.2, shuffle=shuffle, random_state=random_state)

  dates_train = np.sort(dates_train)
  dates_test = np.sort(dates_test)

  x_fret = days[var_list]
  y_fret = days["quant"]

  cnt_varlist =  ['yr','season','elaps','holiday','weekday','workingday','temp','hum','windspeed']
  x_cnt = hours[cnt_varlist + ["hr"] + ["dteday"]+ ["weathersit"] + ["bw", "rbw"]]
  y_cnt = hours["cnt"]

  print()
  x_train = x_fret[days['dteday'].isin(dates_train)]
  x_test = x_fret[days['dteday'].isin(dates_test)]

  y_train = y_fret[days['dteday'].isin(dates_train)]
  y_test = y_fret[days['dteday'].isin(dates_test)]
  #  y_train = MetricData(Wasserstein1D(Wasserstein_dim),y_train)
  #  y_test = MetricData(Wasserstein1D(Wasserstein_dim), y_test)
  y_train = MetricData(Wasserstein1D(Wasserstein_dim), np.c_[[np.array(arr) for arr in y_train.values]])
  y_test  = MetricData(Wasserstein1D(Wasserstein_dim), np.c_[[np.array(arr) for arr in y_test.values]])

  print("x_train", x_train.shape, "x_test", x_test.shape)
  print("y_train", y_train.shape, "  y_test", y_test.shape)
  print()

  x_train_cnt = x_cnt[hours['dteday'].isin(dates_train)]
  x_test_cnt = x_cnt[hours['dteday'].isin(dates_test)]
  y_train_cnt = y_cnt[hours['dteday'].isin(dates_train)]
  y_test_cnt = y_cnt[hours['dteday'].isin(dates_test)]
  print("x_train_cnt", x_train_cnt.shape, "x_test_cnt", x_test_cnt.shape)
  print("y_train_cnt", y_train_cnt.shape, "  y_test_cnt", y_test_cnt.shape)
  print()

  """
    x = hours[var_list]
    y = hours['quant']

    y_metric = MetricData(Wasserstein1D(Wasserstein_dim), np.c_[[np.array(arr) for arr in y.values]])
    y2 = hours['cntday']

    # Test-train split
    random_state = 2
    X_train, X_test, y_train0, y_test0, y_train02, y_test02 = train_test_split(x, y, y2, test_size=0.2, shuffle=False, random_state=random_state)

    x_train = X_train[var_list]
    y_train = MetricData(Wasserstein1D(Wasserstein_dim), np.c_[[np.array(arr) for arr in y_train0.values]])

    x_test = X_test[var_list]
    y_test = MetricData(Wasserstein1D(Wasserstein_dim), np.c_[[np.array(arr) for arr in y_test0.values]])
  """







  model = GlobalFrechet().fit(x_train,y_train)
  pred_test = model.predict(x_test)
  pred_train = model.predict(x_train)
  losses_test = model.y_train_.M.d(y_test, pred_test)
  losses_train = model.y_train_.M.d(y_train, pred_train)

  sorted_test = np.argsort(losses_test)
  sorted_train = np.argsort(losses_train)


  ## Make base plots
  ax_titles = ["0.1 Quantile", "0.5 Quantile", "0.9 Quantile"]
  quantiles_list = [0.1, 0.5, 0.9]
  n_subplots = len(quantiles_list)
  idx_list = make_base_plots("FG", pred_test, y_test, pred_train, y_train, quantiles_list, ax_titles)

  table_RMSE = {}
  table_RMSE.update({"Global Frechet" : [pred_test, y_test, pred_train, y_train]})

  """

  n = 1
  best_idx = sorted_test[:n+1]
  worst_idx = sorted_test[-(n+1):]
  mid_ = int(len(y_test)/2)
  mid_idx = sorted_test[mid_:mid_+n+1]


  #mse_losses_test = mse(y_test.data, pred_test.data)

  #mse_losses_train = mse(y_train, pred_train)


  # code to find IQR error predictions
  sq_err_test = model.y_train_.M.d(y_test, pred_test)**2
  sq_err_train = model.y_train_.M.d(y_train, pred_train)**2

  se_lower_quartile = np.quantile(sq_err_test, q=quantiles_list[0])
  se_median = np.quantile(sq_err_test, q=quantiles_list[1])
  se_upper_quartile = np.quantile(sq_err_test, q=quantiles_list[2])


  idx_se_lower_q = np.argmin(np.abs(sq_err_test - se_lower_quartile))
  idx_se_median_q = np.argmin(np.abs(sq_err_test - se_median))
  idx_se_upper_q = np.argmin(np.abs(sq_err_test - se_upper_quartile))
  idx_list = [idx_se_lower_q, idx_se_median_q, idx_se_upper_q]
  
  print(se_median, sq_err_test[idx_se_median_q])
  print(se_lower_quartile, sq_err_test[idx_se_lower_q])
  print(se_upper_quartile, sq_err_test[idx_se_upper_q])




  figs = []
  figs.extend(plot_QQ(pred_test, y_test, best_idx, "Best Test"))
  figs.extend(plot_QQ(pred_test, y_test, mid_idx,  "Mid Test"))
  figs.extend(plot_QQ(pred_test, y_test, worst_idx, "Worst Test"))
  for fig in figs:
    fig.tight_layout()
    ## dont save these
    #fig.savefig(os.path.join(fig_folder, fig.get_title() + ".pdf"))
    fig.show()

  # ECDF plot 
  fig, ax = plt.subplots()
  i = 12
  x_lin = pred_test.data[i]
  x_lin = np.insert(x_lin, -1, x_lin[-1])
  ys = np.arange(1,Wasserstein_dim+1) / (len(x_lin))#[1:-1]
  ax.hlines(y = ys, xmin=x_lin[:-1], xmax=x_lin[1:],
            color='grey', zorder=1)
  ax.scatter(x_lin[:-1], ys, color='grey', s=3, zorder=2)
  plt.title("eCDF II - " + dates_test[i])
  ## Don't save fig
  # fig.savefig(os.path.join(fig_folder, fig.get_title() + ".pdf"))
  fig.show()


  fig, ax = plt.subplots()
  ax.plot(Wasserstein_grid, pred_test.data[i])
  ax.plot(Wasserstein_grid, y_test.data[i], label = "y")
  plt.title("Quantiles" + dates_test[i])
  plt.legend()
  ## Don't save fig
  # fig.savefig(os.path.join(fig_folder, fig.get_title() + ".pdf"))
  fig.show()


  #staircase
  fig = plt.figure()
  x_lin = pred_test.data[0]
  x_lin = np.insert(x_lin, 0, 0)
  plt.title("eCDF")
  n_ = len(x_lin) 
  cdf = np.arange(1, n_ + 1) / n_
  plt.step(x_lin, cdf, where="post")
  ## Don't save fig
  # fig.savefig(os.path.join(fig_folder, fig.get_title() + ".pdf"))
  fig.show()


  # Discreet pdf
  ys = Wasserstein_grid
  fig = plt.figure()
  x_lin = pred_test.data[0]
  #x = np.where(x_lin == 0, np.finfo(np.float16).eps, x_lin)
  dx = np.diff(x_lin)
  dx = np.where(dx == 0, eps_div,dx)
  pdf_ = np.diff(ys) / dx
  midpoints = (x_lin[1:] + x_lin[:-1]) / 2
  plt.bar(midpoints, pdf_, width=0.1, align="center", edgecolor="black")
  #plt.figure()
  plt.title("discreet pdf")
  #plt.bar(midpoints, pdf_, width=np.diff(x_lin), align="center", edgecolor="black")
  ## Don't save fig
  # fig.savefig(os.path.join(fig_folder, fig.get_title() + ".pdf"))
  fig.show()

  ## NP array from pandas dataframe
  #x_train = x_train.to_numpy()


  ys = Wasserstein_grid
  fig = plt.figure()
  x_lin = pred_test.data[0] #fix
  # = np.where(x_lin == 0, np.finfo(np.float16).eps, x_lin)
  dx = np.diff(x_lin)
  dx = np.where(dx == 0, eps_div, dx)
  pdf_ = np.diff(ys) / dx
  midpoints = (x_lin[1:] + x_lin[:-1]) / 2
  plt.bar(midpoints, pdf_, width=0.1, align="center", edgecolor="black")
  ## Don't save fig
  # fig.savefig(os.path.join(fig_folder, fig.get_title() + ".pdf"))
  fig.show()

  fig=plt.figure()
  plt.bar(midpoints, pdf_, width=1/np.sum(pdf_), align="center", edgecolor="black")
  ## Don't save fig
  # fig.savefig(os.path.join(fig_folder, fig.get_title() + ".pdf"))
  fig.show()




  for i_, i in enumerate(idx_list):
    fig, ax = plt.subplots()
    x_lin = pred_test.data[i]
    x_lin = np.insert(x_lin, -1, x_lin[-1])
    ys = np.arange(1,Wasserstein_dim+1) / (len(x_lin))#[1:-1]
    ax.hlines(y = ys, xmin=x_lin[:-1], xmax=x_lin[1:],
            color='grey', zorder=1)
    ax.scatter(x_lin[:-1], ys, color='grey', s=3, zorder=2)
    plt.title("CDF")
    fig.savefig(os.path.join(fig_folder, "GF CDF " +  str(quantiles_list[i_])+ ".pdf"))
    fig.show()

    ys = Wasserstein_grid
    fig, ax = plt.subplots()
    x_lin = pred_test.data[i]
    #x = np.where(x_lin == 0, np.finfo(np.float16).eps, x_lin)
    dx = np.diff(x_lin)
    dx = np.where(dx == 0, eps_div, dx)
    pdf_ = np.diff(ys) / dx
    midpoints = (x_lin[1:] + x_lin[:-1]) / 2
    ax.bar(midpoints, pdf_, width=0.1, align="center", edgecolor="black")
  #  ax.title("PDF")
    fig.savefig(os.path.join(fig_folder, "GF PDF " +  str(quantiles_list[i_])+ ".pdf"))
    fig.show()    

    fig, ax = plt.subplots()
    ax.plot(pred_test.data[i], y_test[i],  "o", mfc='none' ,color="darkgreen")
    #ax.scatter(pred_test.data[i], y_test[i])
    fig.show()   
    xs = np.linspace(0, np.max(y_test.data[i,:]), 100)
    ys = xs
    ax.plot(xs, ys, "-", color= "lightgreen")
    fig.savefig(os.path.join(fig_folder, "GF QQ " +  str(quantiles_list[i_])+ ".pdf"))
    fig.show()
  """


  #Error plot
  #Wasserstein1D(Wasserstein_dim).d(y_test.data[i], pred_test.data[i])
  fig, ax = plt.subplots(2,1)
  #y_train.data[:,median_idx1]+y_train.data[:,median_idx2]/2
  #median_idx1 = Wasserstein1D(Wasserstein_dim)/2-1
  #median_idx2 = Wasserstein1D(Wasserstein_dim)/2
  ax[0].plot(Wasserstein1D(Wasserstein_dim).d(y_train.data[:,:], pred_train.data[:,:]))
  ax[0].title.set_text("Train")
  #ax[0].set_title('Train and Test', fontsize=10)
  ax[1].plot(Wasserstein1D(Wasserstein_dim).d(y_test.data[:,:], pred_test.data[:,:]))
  ax[1].set_title('Test')
  fig.supxlabel("Days")
  fig.supylabel("Wasserstein distance")
  fig.tight_layout()
  fig.savefig(os.path.join(fig_folder, "GF Error plot.pdf"))
  fig.show()




  ###################################################
  ### COUNT MODEL
  ###################################################

  train_dat = pandas.concat((x_train_cnt, y_train_cnt), axis = 1)
  test_dat = pandas.concat((x_test_cnt, y_test_cnt), axis = 1)


  model_str = "cnt ~   yr + C(hr) *  workingday + weekday + temp + windspeed + hum + C(weathersit)"
  model_str = "cnt ~ C(hr) + yr +  workingday + temp + windspeed + C(bw) + C(rbw)"
  poisglm = smf.glm(model_str, data = train_dat, family=sm.families.Poisson()).fit()
  print(poisglm.summary())

  negbinglm = smf.glm(model_str, data = train_dat, family=sm.families.NegativeBinomial()).fit()
  print(negbinglm.summary())


  # use qaic? for overdispersion
  print( "aic negb  ", negbinglm.aic, "aic pois  ", poisglm.aic) 

  disperse = poisglm.pearson_chi2/poisglm.df_resid
  print(disperse)


  disperse2 = poisglm.deviance/poisglm.df_resid
  print(disperse2)


  pois_pred = poisglm.predict(test_dat)
  nbpred = negbinglm.predict(test_dat)

  pois_pred_train = poisglm.predict(train_dat)
  nbpred_train = negbinglm.predict(train_dat)

      
  def cnt_to_quant(test_dat, pred):
    # Producing quantiles from negative binomial model
    quants = np.zeros((len(np.unique(test_dat["dteday"])), Wasserstein_dim ))
    se_cnt = []
    for j, date in enumerate(np.unique(test_dat["dteday"])):
      for i in range(24):
        se_cnt.append((test_dat[test_dat["dteday"] == date].iloc[i]["cnt"] - pred[test_dat["dteday"] == date].iloc[i])**2)
      quants[j,:] = np.quantile(pred[test_dat["dteday"] == date], Wasserstein_grid)

    return quants, se_cnt


  nb_quants_train, _ = cnt_to_quant(train_dat, nbpred_train)
  pois_quants_train, _ = cnt_to_quant(train_dat, pois_pred_train)


    
  nb_quants, nb_se = cnt_to_quant(test_dat, nbpred)
  nb_mse_cnt = np.mean(nb_se)
  nb_rmse_cnt = np.sqrt(nb_mse_cnt)

  pois_quants, pois_se = cnt_to_quant(test_dat, pois_pred)
  pois_mse_cnt = np.mean(pois_se)
  pois_rmse_cnt = np.sqrt(pois_mse_cnt)

  from sklearn.metrics import mean_squared_error
  print("Mean, y_test_cnt   ", np.mean(y_test_cnt))
  print("RMSE, nbpred   ", np.sqrt(mean_squared_error(y_test_cnt, nbpred)))
  print("RMSE, pois_pred", np.sqrt(mean_squared_error(y_test_cnt, pois_pred)))

  fig, ax1 = plt.subplots()
  #fig.set_title("nbpred vs pois_pred, hours, test")
  ax1.plot(np.arange(len(y_test_cnt)), nbpred, alpha=0.5, label="nbpred")
  ax1.plot(np.arange(len(y_test_cnt)), pois_pred, alpha=0.5, label="pois_pred")
  ax1.plot(np.arange(len(y_test_cnt)), y_test_cnt, alpha=0.5, label="y_test_cnt")
  plt.legend()

  ax2 = ax1.twinx()
  ax2.plot(np.arange(len(y_test_cnt)),np.abs(y_test_cnt - nbpred), label="nbpred abs diff")
  ax2.plot(np.arange(len(y_test_cnt)),np.abs(y_test_cnt - pois_pred), label="pois_pred abs diff")
  ax2.tick_params(axis="y", labelcolor="red")
  ax2.set_xlim([0, 24*7])
  fig.savefig(os.path.join(fig_folder, "cnt_ugly.pdf"))
  fig.show()


  nb_quants = MetricData(Wasserstein1D(Wasserstein_dim), nb_quants)
  pois_quants = MetricData(Wasserstein1D(Wasserstein_dim), pois_quants)


  nb_quants_train = MetricData(Wasserstein1D(Wasserstein_dim), nb_quants_train)
  pois_quants_train = MetricData(Wasserstein1D(Wasserstein_dim), pois_quants_train)

  mse(nb_quants, y_test)



  idx_list = make_base_plots("NB", nb_quants, y_test, nb_quants_train, y_train, quantiles_list, ax_titles)
  idx_list = make_base_plots("POIS", pois_quants, y_test, pois_quants_train, y_train, quantiles_list, ax_titles)

  table_RMSE.update({"Negative Binomial" : [nb_quants, y_test, nb_quants_train, y_train, ]})
  table_RMSE.update({"Poisson" : [pois_quants, y_test, pois_quants_train, y_train,]})



  ###################################################
  ### ADABOOST PLOTS
  ###################################################

  data_train = np.concat((x_train, y_train), axis=1)
  data_test = np.concat((x_test, y_test),axis=1 )


  max_learners = 25
  # Instantiating AdaBoost 
  mywBooster = AdaBoost(
            data = data_train, 
            learner = GlobalFrechet, 
            omega=Wasserstein1D(Wasserstein_dim), 
            t = max_learners,
            dimY = Wasserstein_dim)

  print("Fitting ...")
  mywBooster.fit()
  print("No. learners", mywBooster.t)

  print("betas", mywBooster.betas)
  print("avg loss", mywBooster.betas/(1+mywBooster.betas))

  pred_train = mywBooster.predict(data_train, "mean_fm")
  losses_train_final = mywBooster.omega.d(pred_train[0].data, data_train[:, -Wasserstein_dim:])
  print("Traning loss - ", np.mean(losses_train_final))
  print("Traning loss mse - ", mse(pred_train[0], MetricData(Wasserstein1D(Wasserstein_dim), data_train[:, -Wasserstein_dim:])))


  pred_test = mywBooster.predict(data_test, "mean_fm")
  losses_test_final = mywBooster.omega.d(pred_test[0].data, data_test[:, -Wasserstein_dim:])
  print("Test loss    - ", np.mean(losses_test_final))
  print("Test loss mse - ", mse(pred_test[0], MetricData(Wasserstein1D(Wasserstein_dim), data_test[:, -Wasserstein_dim:])))


  ## Make base plots
  idx_list = make_base_plots("ada", pred_test[0], y_test, pred_train[0], y_train, quantiles_list, ax_titles)
  table_RMSE.update({"AdaBoost" : [pred_test[0], y_test, pred_train[0], y_train ]})


  ## Learners
  losses_train = []
  losses_test = []
  for t in range(mywBooster.t):
      print(t+1, "learner")
      if t == 0:
        ws = None
        losses_train_t = 0
        for i in range(data_train.shape[0]):
          # pred_train_t = MetricData(mywBooster.omega, pred_train[1][:t,i,:]).frechet_mean(ws)
          pred_train_t = MetricData(mywBooster.omega, pred_train[1][t,i,:])
          losses_train_t += mywBooster.omega.d(pred_train_t.data, data_train[i, -Wasserstein_dim:])
        losses_train.append(losses_train_t/data_train.shape[0])

        losses_test_t = 0
        for i in range(data_test.shape[0]):
          pred_test_t = MetricData(mywBooster.omega, pred_test[1][t,i,:])
          losses_test_t += mywBooster.omega.d(pred_test_t.data, data_test[i, -Wasserstein_dim:])
        losses_test.append(losses_test_t/data_test.shape[0])

      else: 
        ws = np.log(1/mywBooster.betas[:t])/np.sum(np.log(1/mywBooster.betas[:t]))
      
        losses_train_t = 0
        for i in range(data_train.shape[0]):
          pred_train_t = MetricData(mywBooster.omega, pred_train[1][:t,i,:]).frechet_mean(ws)
          losses_train_t += mywBooster.omega.d(pred_train_t.data, data_train[i, -Wasserstein_dim:])
        losses_train.append(losses_train_t/data_train.shape[0])

        losses_test_t = 0
        for i in range(data_test.shape[0]):
          pred_test_t = MetricData(mywBooster.omega, pred_test[1][:t,i,:]).frechet_mean(ws)
          losses_test_t += mywBooster.omega.d(pred_test_t.data, data_test[i, -Wasserstein_dim:])
        losses_test.append(losses_test_t/data_test.shape[0])

  fig = plt.figure()
  plt.plot(np.arange(len(losses_train))+1, losses_train, label="train")
  plt.plot(np.arange(len(losses_test))+1, losses_test, label="test")
  plt.legend()
  fig.savefig(os.path.join(fig_folder, "ada_learner.pdf"))
  fig.show()

  ## How weights change for different learners
  ts_to_show = [0, 1, int(mywBooster.t/3), int(mywBooster.t*2/3), mywBooster.t-1]
  ts_to_show = [0, int(mywBooster.t/2), mywBooster.t-1]
  fig = plt.figure()
  plt.title("Learner log weights")
  for t in ts_to_show:
    w = mywBooster.weights[:, t]
    w_ = np.log(w/np.sum(w))
    plt.plot(np.arange(len(mywBooster.weights[:, t])), w_, "-", alpha = 0.4, label=str(t))
    print(t, "weights", mywBooster.weights[:, t].shape, np.sum(mywBooster.weights[:, t]))
  plt.legend()
  fig.savefig(os.path.join(fig_folder, "ada_learner_weights.pdf"))
  fig.show()



  fig = plt.figure()
  plt.title("Learner losses's")
  for t in ts_to_show:
    plt.plot(np.arange(len(mywBooster.losses[:, t])), mywBooster.losses[:, t], "-", alpha = 0.4, label=str(t))
    print(t, "losses", mywBooster.losses[:, t].shape, np.sum(mywBooster.losses[:, t]))
  plt.legend()
  fig.savefig(os.path.join(fig_folder, "ada_learner_losses.pdf"))
  fig.show()




  ## Learner predictions
  #fig, axes = plt.subplots(1, n_subplots, sharey=True)
  #fig.suptitle("Learner predictions")
  for i in range(n_subplots):
  #  ax = axes[i]
    fig, ax = plt.subplots() ##
    fig.suptitle("Learner predictions") ##

    idx = idx_list[i]
    ax.set_title(ax_titles[i] + "\n" + x_test.index[idx])

    cmap = plt.cm.Oranges
  #  cmap = plt.cm.plasma
    norm = plt.Normalize(0,1)
    colors = cmap(norm(np.log(1/mywBooster.betas)))

    ##  plot All learners individually
    ax.plot(Wasserstein_grid, (pred_test[1][0, idx, :]), "-", color=colors[0], label="Individual L")
    for t in range(1, pred_test[1].shape[0]):
      ax.plot(Wasserstein_grid, (pred_test[1][t, idx, :]), "-", color=colors[t])
    ##  Adaboost prediction (combined)
    ax.plot(Wasserstein_grid, (pred_test[0][idx]), "-", color="black", label="Adaboost")

    ##  y_test (target) 
    ax.plot(Wasserstein_grid, (y_test[idx]), "-", color="dodgerblue", label="Data")

    fig.legend() ##
    fig.tight_layout() ##
    fig.savefig(os.path.join(fig_folder, "ada_"+str(quantiles_list[i])+"_learners_predict.pdf"))
    fig.show() ##


  #handles, _ = axes[0].get_legend_handles_labels()
  #labels = ["Individual L", "Adaboost", "Data"]
  #fig.legend(handles, labels, loc="lower center", ncol=len(labels))
  #fig.tight_layout(rect=[0,0.1,1,1])
  #fig.show()



  print("done with predictions for representative quantiles")





  """
  for day in sorted_test[-3:]:
      
      fig = plt.figure()
      plt.plot(y_test.data[day,:], label = "y")
      plt.plot(pred_test.data[day,:], label = "prediction")
      plt.title("Test loss: " + str(Wasserstein1D(Wasserstein_dim).d(y_test.data[day,:], pred_test.data[day,:]))+ " - "+ "Day " +str(day))
      plt.legend()
      fig.show()

      #Q-Q plot
      fig = plt.figure()
      plt.plot(y_test.data[day,:], pred_test.data[day,:], "o", mfc='none' ,color="darkgreen", label = "Q-Q" )
      #xs = np.linspace(0, np.max(np.array(np.max(y.data[day,:]),np.max(pred.data[day,:]))), 100)
      xs = np.linspace(0, np.max(y_test.data[day,:]), 100)

      ys = xs
      plt.plot(xs, ys, "-", color= "lightgreen")
      plt.legend()
      plt.xlabel("Empirical quantiles")
      plt.ylabel("Predicted quantiles")
      plt.title("Test loss: " + str(Wasserstein1D(Wasserstein_dim).d(y_test.data[day,:], pred_test.data[day,:]))+ " - "+ "Day " +str(day))
  """
      

  ##############
  ### MAKE TABLE
  ##############
  print("-"*(2+25+(3+10)*2+2))
  print(f"| {"Model":25} | {"Train":10} | {"Test":10} |")
  for key in table_RMSE:
    pred_test_, y_test_, pred_train_, y_train_ = table_RMSE[key]

    rmse_test_ = np.sqrt(mse(pred_test_, y_test_))
    rmse_train_ = np.sqrt(mse(pred_train_, y_train_))
    print(f"| {key:25} | {rmse_train_:10.3f} | {rmse_test_:10.3f} |")
  print("-"*(2+25+(3+10)*2+2))


  print("final plt.show")
  plt.show()
  print("fin")
