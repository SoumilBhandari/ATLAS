#!/usr/bin/env python3
"""MEP classifier v7 — results only, no plots. 20 features, LOSO, dual thresholds."""
from __future__ import annotations
import argparse, logging, warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger("mep")

# Config
BL_WIN, BLANK_WIN = (-50.,-10.), (-1.,6.)
BP, BP_ORD, NOTCH, NOTCH_Q = (20.,450.), 4, 60., 30.
PTP_WIN, AUC_WIN, PEAK_WIN = (18.,45.), (18.,45.), (10.,60.)
WIDTH_K, EXPECTED_LAT, INNER_FRAC = 3.0, 25.0, 0.2
BP_BANDS = {"lo":(20.,60.), "mid":(60.,150.), "hi":(150.,300.)}
AMP_FEATS = ["ptp_raw","ptp_filt","auc_raw","auc_filt","baseline_rms_raw",
             "baseline_rms_filt","energy_ratio_filt","bp_lo_filt","bp_mid_filt","bp_hi_filt"]
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# Signal processing
def _m(t,w): return (t>=w[0])&(t<=w[1])
def _blank(t,y,w):
    o,m = y.copy(), _m(t,w)
    if not m.any(): return o
    ix = np.flatnonzero(m); o[ix] = np.linspace(o[max(ix[0]-1,0)], o[min(ix[-1]+1,len(y)-1)], len(ix)); return o
def _blc(t,y,w):
    m = _m(t,w); return y-y[m].mean() if m.any() else y.copy()
def _bfilt(fs):
    nyq=fs/2; sos=butter(BP_ORD,[max(.1,BP[0])/nyq,min(BP[1],nyq-1)/nyq],btype="bandpass",output="sos")
    return sos, [iirnotch(w0=NOTCH*h,Q=NOTCH_Q,fs=fs) for h in (1,2,3) if NOTCH*h<nyq]
def _filt(y,sos,n):
    o=sosfiltfilt(sos,y)
    for b,a in n: o=filtfilt(b,a,o)
    return o
def _env(y,w):
    w=max(1,int(w)); n=y.size; cs=np.empty(n+1); cs[0]=0; np.cumsum(y*y,out=cs[1:])
    i=np.arange(n); lo=np.maximum(i+1-w,0); return np.sqrt((cs[i+1]-cs[lo])/(i+1-lo).astype(float))
def _lrun(m):
    b=c=0
    for v in m:
        if v: c+=1; b=max(b,c)
        else: c=0
    return b
def _cpk(e,t):
    c,p=0,False
    for v in(e>t):
        if v and not p: c+=1; p=True
        elif not v: p=False
    return c
def _bp(s,fs,band):
    n=min(len(s),max(16,int(fs*.01)))
    if n<4: return np.nan
    f,p=welch(s,fs=fs,nperseg=n,noverlap=n//2); m=(f>=band[0])&(f<=band[1])
    return float(_trapz(p[m],f[m])) if m.any() else np.nan

# Feature extraction (20)
def _feats(t,raw,filt,er,ef,fs):
    dt=float(np.median(np.diff(t))); mp,ma,mk,mb=_m(t,PTP_WIN),_m(t,AUC_WIN),_m(t,PEAK_WIN),_m(t,BL_WIN); f={}
    for tg,s in[("raw",raw),("filt",filt)]:
        f[f"ptp_{tg}"]=float(np.ptp(s[mp])) if mp.any() else np.nan
        f[f"baseline_rms_{tg}"]=float(np.sqrt((s[mb]**2).mean())) if mb.any() else np.nan
    br,bf = er[mb] if mb.any() else [np.nan], ef[mb] if mb.any() else [np.nan]
    mr,mf = float(np.nanmean(br)), float(np.nanmean(bf))
    sr=max(float(np.nanstd(br,ddof=1)),.05*abs(mr) if np.isfinite(mr) else 0,1e-3)
    sf=max(float(np.nanstd(bf,ddof=1)),.05*abs(mf) if np.isfinite(mf) else 0,1e-3)
    if mk.any():
        pi=np.flatnonzero(mk); ir,iff=int(pi[er[mk].argmax()]),int(pi[ef[mk].argmax()])
        pr,pf=float(er[ir]),float(ef[iff]); f["peak_lat_raw"],f["peak_lat_filt"]=float(t[ir]),float(t[iff])
    else: pr=pf=np.nan; f["peak_lat_raw"]=f["peak_lat_filt"]=np.nan
    f["zpeak_raw"]=(pr-mr)/sr if np.isfinite(pr) else np.nan
    f["zpeak_filt"]=(pf-mf)/sf if np.isfinite(pf) else np.nan
    f["peak_to_base_raw"]=pr/mr if(np.isfinite(pr)and abs(mr)>1e-9)else np.nan
    f["peak_to_base_filt"]=pf/mf if(np.isfinite(pf)and abs(mf)>1e-9)else np.nan
    if ma.any():
        for tg,s,k in[("raw",raw,"baseline_rms_raw"),("filt",filt,"baseline_rms_filt")]:
            d=max(f[k],1e-6) if np.isfinite(f[k]) else 1e-6; f[f"auc_{tg}"]=float(np.abs(s[ma]).sum())*dt/d
    else: f["auc_raw"]=f["auc_filt"]=np.nan
    th=mf+WIDTH_K*sf
    f["width_ms_filt"]=float(_lrun(ef[mk]>th)*dt) if mk.any() else np.nan
    f["lat_dev_filt"]=abs(f["peak_lat_filt"]-EXPECTED_LAT) if np.isfinite(f.get("peak_lat_filt",np.nan)) else np.nan
    f["n_peaks_filt"]=float(_cpk(ef[mk],th)) if mk.any() else np.nan
    if ma.any()and mb.any(): f["energy_ratio_filt"]=float((filt[ma]**2).sum()*dt)/max(float((filt[mb]**2).sum()*dt),1e-9)
    else: f["energy_ratio_filt"]=np.nan
    if mk.any():
        sp=filt[mk]
        for nm,bd in BP_BANDS.items(): f[f"bp_{nm}_filt"]=_bp(sp,fs,bd)
    else:
        for nm in BP_BANDS: f[f"bp_{nm}_filt"]=np.nan
    bl,bh=f.get("bp_lo_filt",np.nan),f.get("bp_hi_filt",np.nan)
    f["bp_ratio_lo_hi_filt"]=bl/max(bh,1e-12) if(np.isfinite(bl)and np.isfinite(bh))else np.nan
    return f

# Load + preprocess
def _load(path):
    df=pd.read_csv(path)
    tc=sorted([c for c in df.columns if _isfloat(c)],key=float)
    t=np.array([float(c) for c in tc],dtype=np.float64); fs=1000./max(float(np.median(np.diff(t))),1e-6)
    dt=float(np.median(np.diff(t))); w=df[tc].apply(pd.to_numeric,errors="coerce").to_numpy(np.float64)
    n=w.shape[0]; sos,nf=_bfilt(fs); rbc,fbc=np.empty_like(w),np.empty_like(w)
    for i in range(n):
        y=w[i]
        if np.any(~np.isfinite(y)): y=pd.Series(y).interpolate(limit_direction="both").to_numpy(np.float64)
        yb=_blank(t,y,BLANK_WIN); rbc[i]=_blc(t,yb,BL_WIN); fbc[i]=_blc(t,_filt(yb,sos,nf),BL_WIN)
    wn=max(1,round(2./dt))
    er=np.vstack([_env(rbc[i],wn) for i in range(n)]); ef=np.vstack([_env(fbc[i],wn) for i in range(n)])
    on=pd.to_numeric(df.get("Onset",np.nan),errors="coerce").to_numpy()
    of=pd.to_numeric(df.get("Offset",np.nan),errors="coerce").to_numpy()
    y=(np.isfinite(on)&np.isfinite(of)&((of-on)>0)).astype(np.int8)
    return t,fs,rbc,fbc,er,ef,y,n
def _isfloat(c):
    try: float(c); return True
    except: return False

# Normalization
def _norm(df,fc):
    amp=[c for c in AMP_FEATS if c in fc and c in df.columns]
    if not amp: return df
    df=df.copy()
    for c in amp:
        g=df.groupby("subject_id")[c]; med=g.transform("median")
        iqr=(g.transform(lambda x:x.quantile(.75))-g.transform(lambda x:x.quantile(.25))).clip(lower=1e-6)
        df[c]=(df[c]-med)/iqr
    return df

# Threshold tuning
def _tune(y,p,mode,tr=.95):
    th=np.linspace(0,1,201)
    if mode=="bal": return float(th[np.argmax([balanced_accuracy_score(y,(p>=t).astype(int)) for t in th])])
    if mode=="det":
        if(y==1).sum()==0 or(y==0).sum()==0: return .5
        bt,bs=0.,-1.
        for t in th:
            pd_=(p>=t).astype(int); tp,fn=((y==1)&(pd_==1)).sum(),((y==1)&(pd_==0)).sum()
            tn,fp=((y==0)&(pd_==0)).sum(),((y==0)&(pd_==1)).sum()
            r,s=tp/(tp+fn)if(tp+fn)else 0, tn/(tn+fp)if(tn+fp)else 0
            if r>=tr and s>bs: bs,bt=s,float(t)
        return bt if bs>=0 else .3
    return .5

def _probs(m,X): return m.predict_proba(X)[:,1] if hasattr(m,"predict_proba") else 1./(1.+np.exp(-m.decision_function(X)))

def _tune_inner(X,y,subs,mdl,mode,tr):
    us=np.unique(subs)
    if len(us)<3: mdl.fit(X,y); return _tune(y,_probs(mdl,X),mode,tr)
    nv=max(1,round(len(us)*INNER_FRAC)); vs=set(np.random.RandomState(42).permutation(us)[:nv])
    vm,tm=np.isin(subs,list(vs)),~np.isin(subs,list(vs))
    if len(np.unique(y[tm]))<2 or len(np.unique(y[vm]))<2: mdl.fit(X,y); return _tune(y,_probs(mdl,X),mode,tr)
    mdl.fit(X[tm],y[tm]); return _tune(y[vm],_probs(mdl,X[vm]),mode,tr)

# Metrics
def _met(yt,yp,ypd):
    cm=confusion_matrix(yt,ypd,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    r={"bal_acc":balanced_accuracy_score(yt,ypd),"f1":f1_score(yt,ypd,zero_division=0),
       "prec":tp/(tp+fp)if(tp+fp)else np.nan,"recall":tp/(tp+fn)if(tp+fn)else np.nan,
       "spec":tn/(tn+fp)if(tn+fp)else np.nan,"fnr":fn/(fn+tp)if(fn+tp)else np.nan,
       "tn":tn,"fp":fp,"fn":fn,"tp":tp}
    if len(np.unique(yt))==2: r["roc_auc"]=roc_auc_score(yt,yp); r["pr_auc"]=average_precision_score(yt,yp)
    else: r["roc_auc"]=r["pr_auc"]=np.nan
    return {k:float(v) for k,v in r.items()}

# Models
def _models(cal=True,rs=0):
    b={"logreg":Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),
        ("c",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=rs))]),
       "rf":Pipeline([("i",SimpleImputer(strategy="median")),
        ("c",RandomForestClassifier(n_estimators=400,class_weight="balanced_subsample",n_jobs=-1,random_state=rs))]),
       "hgb":Pipeline([("i",SimpleImputer(strategy="median")),
        ("c",HistGradientBoostingClassifier(max_depth=4,learning_rate=.1,max_iter=300,min_samples_leaf=20,random_state=rs))])}
    return {k:CalibratedClassifierCV(v,method="isotonic",cv=3) for k,v in b.items()} if cal else b

# LOSO
def _loso(fdf,fc,mdl,mn,tr=.95):
    X,y,subs=fdf[fc].to_numpy(np.float64),fdf["y"].to_numpy(np.int8),fdf["subject_id"].to_numpy(np.int32)
    R={m:{"rows":[],"cm":np.zeros((2,2),int),"thrs":[]} for m in("det","bal")}
    for sid in sorted(np.unique(subs)):
        te,tr_=subs==sid,subs!=sid; Xtr,ytr,Xte,yte=X[tr_],y[tr_],X[te],y[te]
        td=_tune_inner(Xtr,ytr,subs[tr_],mdl,"det",tr); tb=_tune_inner(Xtr,ytr,subs[tr_],mdl,"bal",tr)
        mdl.fit(Xtr,ytr); prob=_probs(mdl,Xte)
        for mode,th in[("det",td),("bal",tb)]:
            r=R[mode]; pred=(prob>=th).astype(np.int8); fm=_met(yte,prob,pred)
            fm.update(subject_id=int(sid),n=int(te.sum()),threshold=th,model=mn)
            r["rows"].append(fm); r["cm"]+=confusion_matrix(yte,pred,labels=[0,1]); r["thrs"].append(th)
        log.info("[%s] s%d | det:thr=%.3f rec=%.3f spec=%.3f | bal:thr=%.3f rec=%.3f spec=%.3f",
                 mn,sid,td,R["det"]["rows"][-1]["recall"],R["det"]["rows"][-1]["spec"],
                 tb,R["bal"]["rows"][-1]["recall"],R["bal"]["rows"][-1]["spec"])
    for m in R: R[m]["df"]=pd.DataFrame(R[m]["rows"]); R[m]["thrs"]=np.array(R[m]["thrs"])
    return R

# Main
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=Path("../data/MEP_Data"))
    p.add_argument("--out-dir",type=Path,default=Path("../outputs"))
    p.add_argument("--models",nargs="+",default=["logreg","rf","hgb"],choices=["logreg","rf","hgb"])
    p.add_argument("--target-recall",type=float,default=.95)
    p.add_argument("--no-calibrate",action="store_true"); p.add_argument("--no-normalize",action="store_true")
    p.add_argument("-v","--verbose",action="store_true"); a=p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,format="%(levelname)s  %(message)s")
    od=a.out_dir; od.mkdir(parents=True,exist_ok=True)
    paths=sorted(a.data_dir.glob("*.csv"),key=lambda p:int(p.stem)); assert paths
    rows=[]
    for path in paths:
        sid=int(path.stem); log.info("Loading %d",sid)
        t,fs,rbc,fbc,er,ef,y,n=_load(path)
        for i in range(n): rows.append({"subject_id":sid,"trial_idx":i,"y":int(y[i]),**_feats(t,rbc[i],fbc[i],er[i],ef[i],fs)})
    fdf=pd.DataFrame(rows); fc=[c for c in fdf.columns if c not in("subject_id","trial_idx","y")]
    if not a.no_normalize: fdf=_norm(fdf,fc)
    fdf.to_csv(od/"features.csv",index=False)
    log.info("%d trials, %d features, pos=%.1f%%",len(fdf),len(fc),100*(fdf.y==1).mean())
    models=_models(cal=not a.no_calibrate); comp=[]
    for mn in a.models:
        log.info("="*40+" %s "+"="*10,mn); R=_loso(fdf,fc,models[mn],mn,a.target_recall)
        for mode in("det","bal"):
            r=R[mode]; r["df"].to_csv(od/f"{mn}_{mode}.csv",index=False)
            tn,fp,fn,tp=r["cm"].ravel(); thrs=r["thrs"]; df=r["df"]
            s={"model":mn,"mode":mode,"agg_tn":int(tn),"agg_fp":int(fp),"agg_fn":int(fn),"agg_tp":int(tp),
               "mean_recall":float(df.recall.mean()),"mean_spec":float(df.spec.mean()),
               "mean_bal_acc":float(df.bal_acc.mean()),"mean_thr":float(thrs.mean()),"std_thr":float(thrs.std())}
            comp.append(s)
            log.info("[%s/%s] fn=%d fp=%d rec=%.3f spec=%.3f bal=%.3f thr=%.3f",mn,mode,fn,fp,s["mean_recall"],s["mean_spec"],s["mean_bal_acc"],s["mean_thr"])
    cdf=pd.DataFrame(comp); cdf.to_csv(od/"comparison.csv",index=False)
    log.info("="*60)
    for _,r in cdf.iterrows(): log.info("%-6s %-3s | rec=%.3f spec=%.3f bal=%.3f | fn=%-5d fp=%-5d | thr=%.3f",r.model,r.mode,r.mean_recall,r.mean_spec,r.mean_bal_acc,r.agg_fn,r.agg_fp,r.mean_thr)

if __name__=="__main__": main()