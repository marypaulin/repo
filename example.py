from matplotlib import pyplot as plt
from osdt_imb_v9 import bbound, predict
import pandas as pd

#file = 'monk1_imbalance2'
#file = 'breast_cancer_wisconsin_binary'
#file = 'diabetes_binary'
#file = 'simulation'
file = 'fourclass_binary1'
#file = 'fico_binary'
df = pd.read_csv('./data/'+file+'.csv')
df = df.values
x = df[:,:-1]
y = df[:,-1]
lamb=0.01
name = 'acc'
w= None
theta = None
leaves_c, pred_c, dic_c, nleaves_c, m_c, n_c, totaltime_c, time_c, R_c, \
COUNT_c, C_c, accu_c, best_is_cart_c, clf_c, \
len_queue, time_queue, time_realize_best_tree, R_best_tree, count_tree= \
bbound(x, y, name, lamb, prior_metric='curiosity', w=w, theta=theta, MAXDEPTH=float('Inf'), 
           MAX_NLEAVES=float('Inf'), niter=float('Inf'), logon=False,
           support=True, incre_support=True, accu_support=False, equiv_points=True,
           lookahead=True, lenbound=True, R_c0 = 1, timelimit=300, init_cart = True,
           saveTree = False, readTree = False)
yhat, out = predict(name, leaves_c, pred_c, nleaves_c, dic_c, x, y, best_is_cart_c, clf_c, w, theta)


plt.plot(time_queue, len_queue)
plt.axvline(x=time_c, color='red', linestyle='--')
plt.ylabel('# of trees in queue')
plt.xlabel('time')
plt.title(' '.join([file, 'time vs # of trees in queue', name]))
plt.savefig('./figure/'+'_'.join([file, 'time_num_queue', name, str(900), 'curiosity'])+'.png', dpi=150)

plt.plot(time_realize_best_tree, R_best_tree)
plt.axhline(y=R_c, color='red', linestyle='--')
plt.axvline(x=time_c, color='red', linestyle='--')
plt.ylabel('risk')
plt.xlabel('time')
plt.title(' '.join([file, 'time vs risk of trees', name]))
plt.savefig('./figure/'+'_'.join([file, 'time_risk', name, str(900), 'curiosity'])+'.png', dpi=150)