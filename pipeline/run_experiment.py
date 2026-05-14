import os
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
import sys, numpy as np, json
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '/home/oper/LabelDiffusion')
exec(open('/home/oper/VLM_IDS/pipeline/vlm_feedback.py').read())

def load_doh(seed, n):
    tr=np.load('/home/oper/DoHBrw2020/npz/doh_train.npz')
    te=np.load('/home/oper/DoHBrw2020/npz/doh_test.npz')
    rng=np.random.default_rng(seed)
    def s(d,n):
        Xl=d['seqs_len'].astype(np.float32); y=d['labels'].astype(int)
        be=rng.permutation(np.where(y==0)[0]); ma=rng.permutation(np.where(y==1)[0])
        n2=min(n//2,len(be),len(ma)); idx=np.concatenate([be[:n2],ma[:n2]]); rng.shuffle(idx)
        return Xl[idx],y[idx]
    return s(tr,n),s(te,2000)

def load_gotham(seed, n):
    d=np.load('/home/oper/LabelDiffusion/data/gotham_seqs_pcap.npz',allow_pickle=True)
    Xl=d['seqs_len'].astype(np.float32); y=d['labels'].astype(int)
    rng=np.random.default_rng(seed)
    be=rng.permutation(np.where(y==0)[0]); ma=rng.permutation(np.where(y==1)[0])
    n2=min(n//2,len(be),len(ma))
    tr=np.concatenate([be[:n2],ma[:n2]]); rng.shuffle(tr)
    n_te=min(1000,len(be)-n2,len(ma)-n2)
    te=np.concatenate([be[n2:n2+n_te],ma[n2:n2+n_te]])
    return (Xl[tr],y[tr]),(Xl[te],y[te])

pipe=VLMFeedbackPipeline(load_vlm=True)
SEEDS=[42,0,1,2,3,4,5,6,7,8]
N_SIZES=[250,500,1000]
all_res={}

for ds_name,load_fn in [('DoH2020',load_doh),('Gotham2025',load_gotham)]:
    all_res[ds_name]={}
    for N in N_SIZES:
        all_res[ds_name][str(N)]={}
        for p in [0.20,0.30,0.40,0.45,0.50]:
            b_l=[]; ct_l=[]; s_l=[]; v_l=[]
            for seed in SEEDS:
                try:
                    (Xl_tr,y_tr),(Xl_te,y_te)=load_fn(seed,N)
                    rng2=np.random.default_rng(seed)
                    y_noisy=y_tr.copy()
                    y_noisy[rng2.random(len(y_tr))<p]^=1
                    Xl_raw=Xl_tr.copy()
                    Xte_raw=Xl_te.copy()
                    sc=StandardScaler()
                    X_s=sc.fit_transform(Xl_tr).astype(np.float32)
                    X_ts=sc.transform(Xl_te).astype(np.float32)
                    res=pipe.run(X_s,y_noisy,y_tr,X_ts,y_te,
                                 noise_rate=p,seed=seed,dataset=ds_name,
                                 X_train_raw=Xl_raw,
                                 X_test_raw=Xte_raw)
                    b_l.append(res['f1_baseline'])
                    ct_l.append(res['f1_ct'])
                    s_l.append(res['f1_standard'])
                    v_l.append(res['f1_vlm'])
                    print('done: {} N={} p={} s={} b={:.4f} ct={:.4f} std={:.4f} vlm={:.4f}'.format(
                        ds_name,N,p,seed,res['f1_baseline'],res['f1_ct'],
                        res['f1_standard'],res['f1_vlm']),flush=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print('err:{}'.format(e),flush=True)
            if b_l:
                all_res[ds_name][str(N)][str(p)]={
                    'baseline': round(float(np.mean(b_l)),4),
                    'ct_only':  round(float(np.mean(ct_l)),4),
                    'standard': round(float(np.mean(s_l)),4),
                    'vlm':      round(float(np.mean(v_l)),4)}

with open('/home/oper/VLM_IDS/results/vlm_full_results.json','w') as f:
    json.dump(all_res,f,indent=2)
print('완료!',flush=True)
