import h5py, glob, os, numpy as np, json
D='/home/kitahara/Desktop/effectiveness_tfcw/reconstruction/dataloaders/data'
def stats(y, bz):
    y=np.asarray(y).reshape(-1).astype(np.int64); n=len(y)
    runs=1+int((np.diff(y)!=0).sum())
    cnt=np.bincount(y.astype(int)); p=cnt[cnt>0]/n
    exp=1+(n-1)*(1-float((p**2).sum()))
    b=[y[i:i+bz] for i in range(0,n,bz)]
    return dict(n=n,classes=int((cnt>0).sum()),sorted=bool((np.diff(y)>=0).all()),runs=runs,expected_runs_if_random=round(exp,1),
                max_run=int(max(len(r) for r in np.split(y,np.flatnonzero(np.diff(y))+1))),bz=bz,
                single_class_batch_frac=round(float(np.mean([len(np.unique(x))==1 for x in b])),4),
                mean_majority_frac=round(float(np.mean([np.bincount(x.astype(int)).max()/len(x) for x in b])),4))
out={}
def lab(fn,keys=('label',)):
    with h5py.File(fn,'r') as f:
        for k in keys:
            if k in f: return f[k][:].reshape(-1)
    raise KeyError(fn)
# ModelNet40 (loader: glob order, concatenated)
for part in ('test','train'):
    fs=glob.glob(f'{D}/modelnet40_ply_hdf5_2048/ply_data_{part}*.h5')
    y=np.concatenate([lab(f) for f in fs])
    out[f'modelnet40_{part}']=stats(y,32); out[f'modelnet40_{part}']['files']=[os.path.basename(f) for f in fs]
for c in ('clean','jitter_0','jitter_1','jitter_2','jitter_3','jitter_4','add_global_0','add_global_1','add_global_2','add_global_3','add_global_4'):
    out[f'modelnet_c/{c}']=stats(lab(f'{D}/modelnet_c/{c}.h5'),32)
for c in ('clean','jitter_0','jitter_1','jitter_2','jitter_3','jitter_4','add_global_0','add_global_1','add_global_2','add_global_3','add_global_4'):
    fn=f'{D}/shapenet_c/{c}.h5'
    with h5py.File(fn,'r') as f: keys=list(f.keys()); shp=f['data'].shape
    out[f'shapenet_c/{c}']=stats(lab(fn),128); out[f'shapenet_c/{c}']['keys']=keys; out[f'shapenet_c/{c}']['data_shape']=list(shp)
json.dump(out,open('/home/kitahara/Desktop/effectiveness_tfcw/forensics_m1_scan/order_diag/grouping.json','w'),indent=1)
for k,v in out.items(): print(k,{a:b for a,b in v.items() if a not in('files',)})
