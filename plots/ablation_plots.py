import seaborn as sns
import pandas
import matplotlib.pyplot as plt

sns.set()
objs_data = pandas.DataFrame({'Unique Objects': [100,100,400,400,800,800,1600,1600], 
                              'vals': [0.468,0.629,0.482,0.620,0.466,0.614,0.516,0.647], 
                              'type': ['AP', 'AR', 'AP', 'AR', 'AP', 'AR', 'AP', 'AR']})
g = sns.catplot(x="type", y="vals", hue="Unique Objects", kind="bar", data=objs_data)
g.set_axis_labels('', '')
plt.savefig('/home/mjd3/Pictures/seg_fig/objs_ablation.pdf', dpi=300, bbox_inches='tight')

ims_data = pandas.DataFrame({'Training Images': [4000,4000,8000,8000,20000,20000,40000,40000], 
                              'vals': [0.460,0.593,0.484,0.615,0.486,0.649,0.516,0.647], 
                              'type': ['AP', 'AR', 'AP', 'AR', 'AP', 'AR', 'AP', 'AR']})
g = sns.catplot(x="type", y="vals", hue="Training Images", kind="bar", data=ims_data)
g.set_axis_labels('', '')
plt.savefig('/home/mjd3/Pictures/seg_fig/ims_ablation.pdf', dpi=300, bbox_inches='tight')
