# ModelNet40 plain t-FCW

Faithful historical entry point: `bash scripts/modelnet.sh` from the external `reconstruction/` source directory. The original script sets surface `knnxyz`, K `[66,66,66,66]`, point hierarchy `[1024,512,256,128]`, seed 3407 through the runner, and searches γ over 0,2,…,9998 using labeled test data. The [result](../../results/modelnet40/summary.json) is 85.1297% at γ6090 versus paper 84.8%. Historical source is not duplicated here because its redistribution license is unresolved.
