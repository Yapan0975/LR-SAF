# coding: utf-8
"""Unified STANDARD dataset-level LCNN sAP for the main YorkUrban table.
AFM / DeepLSD / HAWPv2 / LR-SAF, all on the same val-22 split, each ranked
by its native confidence. Replaces the buggy per-image lenient evaluator."""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
ROOT='/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT+'/code/lr_saf')
sys.path.insert(0, ROOT+'/code/afm_baseline')
sys.path.insert(0, ROOT+'/code/afm_baseline/lib')
os.chdir(ROOT+'/code/afm_baseline')
import torch, numpy as np, cv2
from config import cfg; cfg.merge_from_file('experiments/afm_atrous.yaml')
from modeling.net import build_network
from lib.squeeze_to_lsg import lsgenerator
from data import YorkUrbanSubset
from model import build_lr_saf
from confidence_head import ConfidenceMLP, compute_segment_features
from metrics import f_measure, image_records, sap_dataset
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
THS=(5,10,15)
seed=42; torch.manual_seed(seed); np.random.seed(seed)
ds=YorkUrbanSubset(in_res=320)
rng=np.random.RandomState(seed); perm=rng.permutation(len(ds)); val_idx=perm[80:].tolist()
items=[]
for vi in val_idx:
    it=ds[vi]; H_o,W_o=it['H_orig'],it['W_orig']
    gt=it['lines'].numpy().copy(); gt[:,0::2]*=W_o/320.0; gt[:,1::2]*=H_o/320.0
    items.append({'name':it['name'],'gt':gt,'H':H_o,'W':W_o})
names=set(x['name'] for x in items)
print('val',len(items),'images')

def fmeas(pred,gt,H,W):
    DS=4; Hd=max(H//DS,32); Wd=max(W//DS,32)
    kp=pred.copy(); kp[:,0::2]/=DS; kp[:,1::2]/=DS
    gp=gt.copy(); gp[:,0::2]/=DS; gp[:,1::2]/=DS
    return f_measure(kp,gp,Hd,Wd)['F']

res={}

# ---- AFM ----
afm=build_network(cfg).cuda().eval()
afm.load_state_dict(torch.load(ROOT+'/checkpoints/atrous/weight/model_final.pth.tar',map_location='cuda',weights_only=False),strict=True)
recs=[]; fs=[]
for x in items:
    img=cv2.imread(os.path.join(ds.root,x['name'],x['name']+'.jpg'))
    xi=cv2.resize(img,(320,320)).astype(np.float32)/255.0; xi=(xi-MEAN)/STD
    xt=torch.from_numpy(xi).permute(2,0,1).unsqueeze(0).cuda()
    with torch.no_grad(): out=afm(xt)
    off=(out[0] if isinstance(out,(list,tuple)) else out)[0].cpu().numpy().astype(np.float32)
    lines,_,_=lsgenerator(off); lines=np.asarray(lines)
    if len(lines)==0: recs.append(image_records(np.zeros((0,4),np.float32),np.zeros(0,np.float32),x['gt'],x['H'],x['W'])); fs.append(0.0); continue
    kept=lines[:,:4].copy(); kept[:,0::2]*=x['W']/320.0; kept[:,1::2]*=x['H']/320.0
    sc=1.0/(lines[:,4]+1e-3) if lines.shape[1]>=5 else np.ones(len(lines))
    recs.append(image_records(kept,sc,x['gt'],x['H'],x['W'])); fs.append(fmeas(kept,x['gt'],x['H'],x['W']))
ap=sap_dataset(recs,THS); res['AFM']={'F':float(np.mean(fs)),**{f'sAP{t}':ap[t] for t in THS}}
print('AFM',res['AFM'])

# ---- LR-SAF + geom head ----
lr=build_lr_saf(device='cuda').eval()
lr.load_state_dict(torch.load(ROOT+'/checkpoints/lr_saf_best.pth',map_location='cuda',weights_only=False)['model'],strict=True)
gh=ConfidenceMLP(in_dim=9,hidden=64).cuda().eval()
gh.load_state_dict(torch.load(ROOT+'/checkpoints/lr_saf_conf_head.pth',weights_only=False)['head'])
recs=[]; fs=[]
for x in items:
    img=cv2.imread(os.path.join(ds.root,x['name'],x['name']+'.jpg'))
    xi=cv2.resize(img,(320,320)).astype(np.float32)/255.0; xn=(xi-MEAN)/STD
    xt=torch.from_numpy(xn).permute(2,0,1).unsqueeze(0).cuda()
    with torch.no_grad(): o=lr(xt)
    off=o['a'][0].cpu().numpy().astype(np.float32); junc=o['junc'][0].cpu().numpy().astype(np.float32)
    lines,_,_=lsgenerator(off); lines=np.asarray(lines)
    if len(lines)==0: recs.append(image_records(np.zeros((0,4),np.float32),np.zeros(0,np.float32),x['gt'],x['H'],x['W'])); fs.append(0.0); continue
    geom=compute_segment_features(lines,off,junc,H=320,W=320).cuda()
    sc=gh(geom).detach().cpu().numpy()
    kept=lines[:,:4].copy(); kept[:,0::2]*=x['W']/320.0; kept[:,1::2]*=x['H']/320.0
    recs.append(image_records(kept,sc,x['gt'],x['H'],x['W'])); fs.append(fmeas(kept,x['gt'],x['H'],x['W']))
ap=sap_dataset(recs,THS); res['LR-SAF']={'F':float(np.mean(fs)),**{f'sAP{t}':ap[t] for t in THS}}
print('LR-SAF',res['LR-SAF'])

# ---- HAWPv2 (cached preds) ----
try:
    hp=json.load(open(ROOT+'/code/hawp_baseline/checkpoints/york_test.json'))
    gtmap={x['name']:x for x in items}
    recs=[]; fs=[]
    for p in hp:
        fn=p['filename'].split('.')[0]
        if fn not in gtmap: continue
        x=gtmap[fn]; pl=np.array(p['lines_pred'],np.float32); sc=np.array(p['lines_score'],np.float32)
        recs.append(image_records(pl,sc,x['gt'],x['H'],x['W'])); fs.append(fmeas(pl,x['gt'],x['H'],x['W']))
    ap=sap_dataset(recs,THS); res['HAWPv2']={'F':float(np.mean(fs)),**{f'sAP{t}':ap[t] for t in THS},'n':len(recs)}
    print('HAWPv2',res['HAWPv2'])
except Exception as e:
    print('HAWP failed:',e)

# ---- DeepLSD ----
try:
    sys.path.insert(0, ROOT+'/code/deeplsd_baseline')
    from deeplsd.models.deeplsd_inference import DeepLSD
    conf={'detect_lines':True,'line_detection_params':{'merge':False,'filtering':True,'grad_nfa':True,'grad_thresh':3}}
    dm=DeepLSD(conf).cuda().eval()
    dm.load_state_dict(torch.load(ROOT+'/code/deeplsd_baseline/weights/deeplsd_wireframe.tar',map_location='cuda',weights_only=False)['model'],strict=False)
    recs=[]; fs=[]
    for x in items:
        img=cv2.imread(os.path.join(ds.root,x['name'],x['name']+'.jpg'))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        t=torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad(): o=dm({'image':t})
        ln=o['lines'][0]; ln=ln.cpu().numpy() if isinstance(ln,torch.Tensor) else np.asarray(ln)
        if len(ln)==0: recs.append(image_records(np.zeros((0,4),np.float32),np.zeros(0,np.float32),x['gt'],x['H'],x['W'])); fs.append(0.0); continue
        pl=ln.reshape(-1,4)
        sc=o.get('line_scores',[None])[0] if 'line_scores' in o else None
        sc=(sc.cpu().numpy() if isinstance(sc,torch.Tensor) else (np.asarray(sc) if sc is not None else np.ones(len(pl))))
        if sc is None or len(sc)!=len(pl): sc=np.ones(len(pl))
        recs.append(image_records(pl,sc,x['gt'],x['H'],x['W'])); fs.append(fmeas(pl,x['gt'],x['H'],x['W']))
    ap=sap_dataset(recs,THS); res['DeepLSD']={'F':float(np.mean(fs)),**{f'sAP{t}':ap[t] for t in THS}}
    print('DeepLSD',res['DeepLSD'])
except Exception as e:
    print('DeepLSD failed:',e)

json.dump(res,open(ROOT+'/logs/main_table_sap.json','w'),indent=2)
print('SAVED logs/main_table_sap.json')
