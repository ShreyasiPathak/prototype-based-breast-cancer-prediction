import matplotlib.pyplot as plt

#------------------------ Align loss tuning --------------#
x_alignwt = [2, 3, 4, 5]
y_best_auc = [0.81, 0.80, 0.81, 0.82]
y_best_auc_proto = [98, 32, 182, 78]
y_last_auc = [0.78, 0.76, 0.77, 0.78]
y_final_proto = [30, 32, 34, 34]

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x_alignwt, y_best_auc, label='Best performing epoch')
ax1.plot(x_alignwt, y_last_auc, label='Last epoch')

ax2.plot(x_alignwt, y_best_auc_proto, label='Best performing epoch')
ax2.plot(x_alignwt, y_final_proto, label='Last epoch')

plt.xlabel('Alignment loss weight')
ax1.set_ylabel('AUC')
ax2.set_ylabel('# non-zero prototypes')
ax1.legend()
ax2.legend()
ax2.set_ylim(0, 200)
plt.xticks([2,3,4,5])
plt.savefig('./alignlosswt_tuning.pdf', bbox_inches='tight', format="pdf")
plt.show()

#------------------------ Classification loss tuning --------------#
x_classwt = [2, 3, 4, 5]
y_best_auc = [0.81, 0.79, 0.82, 0.82]
y_best_auc_proto = [98, 37, 94, 99]
y_last_auc = [0.78, 0.77, 0.77, 0.78]
y_final_proto = [30, 33, 38, 40]

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x_classwt, y_best_auc, label='Best performing epoch')
ax1.plot(x_classwt, y_last_auc, label='Last epoch')

ax2.plot(x_classwt, y_best_auc_proto, label='Best performing epoch')
ax2.plot(x_classwt, y_final_proto, label='Last epoch')

plt.xlabel('Classification loss weight')
ax1.set_ylabel('AUC')
ax2.set_ylabel('# non-zero prototypes')
ax1.legend()
ax2.legend()
ax2.set_ylim(0, 200)
plt.xticks([2,3,4,5])
plt.savefig('./classificationlosswt_tuning.pdf', bbox_inches='tight', format="pdf")
plt.show()

#------------------------ Tanh loss tuning --------------#
x_tanlwt = [2, 3, 4, 5]
y_best_auc = [0.81, 0.81, 0.82, 0.82]
y_best_auc_proto = [98, 91, 92, 96]
y_last_auc = [0.78, 0.77, 0.77, 0.76]
y_final_proto = [30, 31, 30, 26]

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x_tanlwt, y_best_auc, label='Best performing epoch')
ax1.plot(x_tanlwt, y_last_auc, label='Last epoch')

ax2.plot(x_tanlwt, y_best_auc_proto, label='Best performing epoch')
ax2.plot(x_tanlwt, y_final_proto, label='Last epoch')

plt.xlabel('Tanh loss weight')
ax1.set_ylabel('AUC')
ax2.set_ylabel('# non-zero prototypes')
ax1.legend()
ax2.legend()
ax2.set_ylim(0, 200)
plt.xticks([2,3,4,5])
plt.savefig('./tanhlosswt_tuning.pdf', bbox_inches='tight', format="pdf")
plt.show()