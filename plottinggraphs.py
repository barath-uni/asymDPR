import matplotlib.pyplot as plt
import numpy as np
# def hist_and_lines():
#     np.random.seed(0)
#     fig, ax = plt.subplots(1, 2, figsize=(11, 4))
#     # ax[0].plot(np.random.rand(10))
#     # ax[1].legend(['KG-BERT', 'RMPI', 'SIMPATH'])
#     for i in range(3):
#         print(i)
#         if i==0:
#             data=np.array([[1000,20],[2000,500], [3000,1200],[4000,2200]])
#             # ax[0].plot(np.random.rand(7))   
#             ax[0].plot(*data.T)
#             # ax[0].plot([20, 500, 1200, 2200, 0, 0])
#         if i==1:
#             data=np.array([[1000,10],[2000,100], [3000,1000],[4000,1500]])
#             ax[0].plot(*data.T)
#             # ax[0].plot([10, 10, 10, 10])
#         if i==2:
#             data=np.array([[1000,10],[2000,100], [3000,300],[4000,400]])
#             ax[0].plot(*data.T)
#             # ax[0].plot([120, 500, 700, 1200])
#         # ax[0].plot(np.random.rand(10))
#     # ax[0].set_xlim([1000, 7000])
    
#     ax[0].set_ylabel('Training Time (secs)')
#     ax[0].set_xlabel('# of Training Triples')
#     # ax[1].set_xlim([1000, 4000])

#     for i in range(3):

#         if i==0:
#             data=np.array([[1000,500],[2000,500], [3000,500],[4000,500]])
#             # ax[0].plot(np.random.rand(7))
#             ax[1].plot(*data.T)
#             # ax[0].plot([20, 500, 1200, 2200, 0, 0])
#         if i==1:
#             data=np.array([[1000,200],[2000,200], [3000,200],[4000,200]])
#             ax[1].plot(*data.T)
#             # ax[0].plot([10, 10, 10, 10])
#         if i==2:
#             data=np.array([[1000,10],[2000,10], [3000,10],[4000,10]])
#             ax[1].plot(*data.T)
#     ax[1].set_ylabel('Test Time/1000 triples (secs)')
#     ax[1].set_xlabel('# of Training Triples')
#     ax[1].legend(['KG-BERT', 'RMPI', 'SIMPATH'])
# x = np.asarray([12, 8, 4, 2, 0])
# y = np.asarray([0.9386, 0.4322, 0.2214, 0.1245, 0.0512])

# with plt.style.context('fivethirtyeight'):
#     # plt.ylabel('Training Time (secs)', fontsize=16)
#     # plt.xlabel('# of Training triples', fontsize=16)
#     # hist_and_lines()
#     hist_and_lines()
#     plt.title("Training and Inference Time of models")
#     # plt.legend()
#     plt.show()
import pandas as pd

def plot_clustered_stacked(dfall, labels=None, title="Relation Type for each relation in pre-processed FB15K-237",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    return axe

fb15k237sbert = [
    [164, 0, 8, 0],
    [52, 0, 2, 0],
    [10, 0, 1, 0]
    
]

fb15k237tfid = [
    [209, 0, 9, 0],
    [5, 0, 2, 0],
    [12, 0, 0, 0]
]
# create fake dataframes
df1 = pd.DataFrame(fb15k237tfid,
                   index=["Train", "Dev", "Test"],
                   columns=["N-N", "1-N", "N-1", "1-1"])


df2 = pd.DataFrame(fb15k237sbert,
                   index=["Train", "Dev", "Test",],
                   columns=["N-N", "1-N", "N-1", "1-1"])

plt.rcParams["figure.figsize"] = (22,3)
with plt.style.context('bmh'):

    # Then, just call :
    plot_clustered_stacked([df1, df2],["FB15K-237-TFIDF", "FB15K-237-SBERT"])
    plt.show()