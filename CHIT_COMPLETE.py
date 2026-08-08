# ============================================================
# CHIT MCAR/MAR/MNAR — TAM ANALİZ KODU
# Repository: https://github.com/kurbankotan/chit-mcar-mar-mnar
#
# GEREKSINIMLER:
#   - Colab Pro veya Pro+ (High-RAM + A100 GPU önerilir)
#   - CKD  (~15 dk),  HDD (~45 dk),  MPED (~6-8 saat)
#
# ÇIKIŞ (Drive'a otomatik kaydedilir):
#   chit_complete_mcar.json / mar / mnar
#   chit_complete_hdd_mcar.json / mar / mnar
#   chit_complete_mped_mcar.json / mar / mnar
# ============================================================

from google.colab import drive, files
drive.mount('/content/drive')

import pandas as pd, numpy as np, json, time, gc, warnings, shutil, os
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, mean_squared_error)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               HistGradientBoostingRegressor)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from scipy import stats as scipy_stats

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU: {[g.name for g in gpus]}")
else:
    print("GPU bulunamadı, CPU kullanılacak")

# ── Veri yolları ─────────────────────────────────────────────────────────────
CKD_PATH  = '/content/drive/MyDrive/Colab Notebooks/datasets/kidney_disease/kidney_disease.csv'
HDD_PATH  = '/content/drive/MyDrive/Colab Notebooks/datasets/Heart Disease Dataset/heart.csv'
MPED_PATH = '/content/drive/MyDrive/Colab Notebooks/datasets/Mice Protein Expression Dataset/Data_Cortex_Nuclear.csv'
DRIVE_OUT = '/content/drive/MyDrive/Colab Notebooks/'

# ── Veri yükleme fonksiyonları ───────────────────────────────────────────────
def load_ckd():
    df = pd.read_csv(CKD_PATH)
    df.set_index('id', inplace=True)
    df.rename(columns={'classification': 'class'}, inplace=True)
    df['dm']    = df['dm'].astype(str).str.strip().replace({'\\tno':'no','\\tyes':'yes',' yes':'yes'})
    df['cad']   = df['cad'].astype(str).str.strip().replace({'\\tno':'no'})
    df['class'] = df['class'].astype(str).str.strip().replace({'ckd\\t':'ckd'})
    df['pcv']   = df['pcv'].astype(str).str.strip().replace({'\\t?':np.nan,'\\t43':'43'})
    df['wc']    = df['wc'].astype(str).str.strip().replace({'\\t?':np.nan,'\\t6200':'6200','\\t8400':'8400'})
    df['rc']    = df['rc'].astype(str).str.strip().replace({'\\t?':np.nan,'\\t3.9':'3.9','\\t4.0':'4.0','\\t5.2':'5.2'})
    df['class'] = df['class'].map({'ckd': 1, 'notckd': 0})
    for col in ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]:
        s = df[col].astype(str).str.strip().str.lower().replace({'nan':np.nan,'none':np.nan,'nat':np.nan})
        df[col] = s.map({v: float(i) for i, v in enumerate(sorted(s.dropna().unique()))})
    for col in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_hdd():
    df = pd.read_csv(HDD_PATH)
    df = df.replace('?', np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'target' in df.columns:
        df = df.rename(columns={'target': 'class'})
    return df

def load_mped():
    le = LabelEncoder()
    df = pd.read_csv(MPED_PATH)
    if 'MouseID' in df.columns:
        df = df.drop('MouseID', axis=1)
    df['class'] = le.fit_transform(df['class'].astype(str))
    for col in ['Genotype', 'Treatment', 'Behavior']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ── Mekanizma simülasyonları ─────────────────────────────────────────────────
def sim_mcar(df, rate=0.1):
    mask = np.random.rand(*df.shape) < rate
    mask[:, df.columns.get_loc('class')] = False
    d = df.copy(); d[mask] = np.nan; return d

def sim_mnar(df, col, thr=None, quantile=None):
    d = df.copy()
    if thr is not None:
        d.loc[d[col] < thr, col] = np.nan
    elif quantile is not None:
        d.loc[d[col] < d[col].quantile(quantile), col] = np.nan
    return d

# ── CHIT normalizasyon ───────────────────────────────────────────────────────
def normalize_for_chit(df_sim):
    cc = df_sim['class'].copy()
    ft = df_sim.drop('class', axis=1)
    out = pd.DataFrame(StandardScaler().fit_transform(ft),
                       columns=ft.columns, index=df_sim.index)
    out['class'] = cc
    return out

# ── CHIT hızlandırılmış (sütun bazlı) ───────────────────────────────────────
def chit_fast(df_norm, col_method='mean', row_model='rf', seed=42):
    feat_cols = [c for c in df_norm.columns if c != 'class']
    X_all = df_norm[feat_cols].copy()
    class_col = df_norm['class'].copy()
    Y = X_all[~X_all.isnull().any(axis=1)].copy()
    miss_cols = sorted(X_all.columns[X_all.isnull().any(axis=0)].tolist(),
                       key=lambda c: X_all[c].isnull().sum(), reverse=True)
    X_filled = X_all.copy()
    for col in miss_cols:
        other = [c for c in feat_cols if c != col]
        Yv = Y.dropna(subset=[col] + other)
        if   col_method == 'mean':   prov = Yv[col].mean()    if len(Yv) > 0 else 0.0
        elif col_method == 'median': prov = Yv[col].median()  if len(Yv) > 0 else 0.0
        elif col_method == 'mod':    prov = Yv[col].mode()[0] if len(Yv) > 0 else 0.0
        elif col_method == 'fwd':    prov = Yv[col].iloc[-1]  if len(Yv) > 0 else 0.0
        elif col_method == 'bwd':    prov = Yv[col].iloc[0]   if len(Yv) > 0 else 0.0
        else:                        prov = Yv[col].mean()    if len(Yv) > 0 else 0.0
        if pd.isnull(prov): prov = 0.0
        miss_idx = X_filled.index[X_filled[col].isnull()]
        if len(miss_idx) == 0: continue
        X_temp = X_filled.copy()
        for oc in other:
            if X_temp[oc].isnull().any():
                fv = Y[oc].mean() if not pd.isnull(Y[oc].mean()) else 0.0
                X_temp[oc] = X_temp[oc].fillna(fv)
        if len(Yv) >= 5:
            try:
                if   row_model == 'rf':  reg = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
                elif row_model == 'knn': reg = KNeighborsRegressor(n_neighbors=min(5, len(Yv)))
                elif row_model == 'dt':  reg = DecisionTreeRegressor(random_state=seed)
                elif row_model == 'lr':  reg = LinearRegression()
                elif row_model == 'svr': reg = SVR(C=1.0)
                elif row_model == 'hgb': reg = HistGradientBoostingRegressor(random_state=seed)
                elif row_model == 'dl':  reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=seed)
                else:                    reg = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
                reg.fit(Yv[other].values, Yv[col].values)
                X_filled.loc[miss_idx, col] = reg.predict(X_temp.loc[miss_idx, other].values)
            except:
                X_filled.loc[miss_idx, col] = prov
        else:
            X_filled.loc[miss_idx, col] = prov
        newly = X_filled.loc[miss_idx][~X_filled.loc[miss_idx].isnull().any(axis=1)]
        if len(newly) > 0:
            Y = pd.concat([Y, newly], ignore_index=True)
    X_filled['class'] = class_col
    return X_filled

# ── 42 konfigürasyon ─────────────────────────────────────────────────────────
COL_METHODS = [('mean','Ortalama'), ('median','Medyan'), ('mod','Mod'),
               ('fwd','Ardısık'),   ('knn','KNN Imp'),   ('bwd','Interpolate')]
ROW_MODELS  = [('rf','Random Forest'), ('knn','K-Nearest N'), ('dt','Decision Tree'),
               ('lr','Linear Reg'),    ('svr','SVR'),          ('hgb','Grad Boost'),
               ('dl','Deep Learn')]

# ── GridSearchCV parametreleri ───────────────────────────────────────────────
param_KNN = {'n_neighbors': list(range(1,11)), 'weights': ["uniform","distance"], "leaf_size": [10,20,30,40,50]}
param_LR  = {"C": np.logspace(-4,4,50), "penalty": ['l1','l2'], "random_state": [0,1,2,3,4,5]}
param_SVC = {'C': [0.1,1,2,3,4,5,6,7,8,9,10,100], 'gamma': [1,0.1,0.01,0.001]}
param_DT  = {'criterion': ['gini','entropy'], 'max_depth': [2,4,6,8,10,12]}
param_RFC = {'bootstrap': [True,False], 'max_depth': [10,20,None], 'max_features': ['sqrt'],
             'min_samples_leaf': [1,2,4], 'min_samples_split': [2,5,10], 'n_estimators': [10,100]}
param_GNB = {'var_smoothing': np.logspace(0,-9,num=100)}
param_MLP = {'solver': ["lbfgs","sgd","adam"], 'alpha': [1e-5], 'activation': ['relu','logistic','tanh']}

# ── DL (Keras, GPU, Dropout kaldırıldı) ─────────────────────────────────────
def run_dl(X_tr, y_tr, X_te, y_te):
    tf.keras.backend.clear_session()
    y_tr_int = np.array(y_tr, dtype=np.int32)
    y_te_int = np.array(y_te, dtype=np.int32)
    n_cls    = int(np.max(y_tr_int)) + 1
    y_tr_cat = tf.keras.utils.to_categorical(y_tr_int, num_classes=n_cls)
    y_te_cat = tf.keras.utils.to_categorical(y_te_int, num_classes=n_cls)
    device   = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    with tf.device(device):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1000, activation='relu', input_shape=(X_tr.shape[1],)),
            tf.keras.layers.Dense(500,  activation='relu'),
            tf.keras.layers.Dense(300,  activation='relu'),
            tf.keras.layers.Dense(n_cls, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=['accuracy'])
        history = model.fit(X_tr.astype(np.float32), y_tr_cat,
                            validation_data=(X_te.astype(np.float32), y_te_cat),
                            batch_size=20, epochs=10, verbose=0)
        pred = np.argmax(model.predict(X_te.astype(np.float32), verbose=0), axis=1)
    tf.keras.backend.clear_session(); gc.collect()
    return {
        'acc':  round(accuracy_score(y_te_int, pred), 4),
        'f1':   round(f1_score(y_te_int, pred, average='macro', zero_division=0), 4),
        'prec': round(precision_score(y_te_int, pred, average='macro', zero_division=0), 4),
        'rec':  round(recall_score(y_te_int, pred, average='macro', zero_division=0), 4),
        'pred': pred.tolist(), 'yte': y_te_int.tolist(),
        'val_acc_history':   history.history['val_accuracy'],
        'train_acc_history': history.history['accuracy'],
    }

# ── Sınıflandırıcılar ────────────────────────────────────────────────────────
def run_classifiers(X_tr, y_tr, X_te, y_te, num_classes):
    res = {}
    for name, clf, params in [
        ('KNN', KNeighborsClassifier(), param_KNN),
        ('LR',  LogisticRegression(max_iter=500, solver='saga'), param_LR),
        ('SVC', SVC(), param_SVC),
        ('DT',  DecisionTreeClassifier(), param_DT),
        ('RFC', RandomForestClassifier(), param_RFC),
        ('GNB', GaussianNB(), param_GNB),
        ('MLP', MLPClassifier(max_iter=500), param_MLP),
    ]:
        try:
            g = GridSearchCV(clf, params, cv=2, scoring='accuracy', n_jobs=-1)
            g.fit(X_tr, y_tr)
            pred = g.best_estimator_.predict(X_te)
            res[name] = {
                'acc':  round(accuracy_score(y_te, pred), 4),
                'f1':   round(f1_score(y_te, pred, average='macro', zero_division=0), 4),
                'prec': round(precision_score(y_te, pred, average='macro', zero_division=0), 4),
                'rec':  round(recall_score(y_te, pred, average='macro', zero_division=0), 4),
                'pred': pred.tolist(), 'yte': list(y_te),
            }
        except Exception as e:
            print(f"    {name} hata: {e}")
            res[name] = {'acc':0.0,'f1':0.0,'prec':0.0,'rec':0.0,'pred':[],'yte':list(y_te)}
    try:
        dl = run_dl(X_tr, y_tr, X_te, y_te)
        res['DL'] = dl
    except Exception as e:
        print(f"    DL hata: {e}")
        res['DL'] = {'acc':0.0,'f1':0.0,'prec':0.0,'rec':0.0,'pred':[],'yte':list(y_te)}
    return res

# ── İstatistiksel hesaplamalar ───────────────────────────────────────────────
def wci(acc, n=80, z=1.96):
    if acc <= 0: return (0.0, 0.0)
    if acc >= 1: return (1.0, 1.0)
    d = 1 + z**2/n; c = (acc + z**2/(2*n))/d
    m = z * np.sqrt(acc*(1-acc)/n + z**2/(4*n**2))/d
    return (round(max(0, c-m), 4), round(min(1, c+m), 4))

def mcnemar_test(y, p1, p2):
    b = int(np.sum((p1==y) & (p2!=y)))
    c = int(np.sum((p1!=y) & (p2==y)))
    if b + c == 0: return 1.0, b, c
    chi2 = (abs(b-c) - 1)**2 / (b+c)
    return round(float(scipy_stats.chi2.sf(chi2, df=1)), 6), b, c

# ── RMSE hesaplama ───────────────────────────────────────────────────────────
def compute_rmse(df_sim, target_col, mask_rows, df_orig, scaler_ref):
    if len(mask_rows) == 0:
        return {'n': 0, 'CHIT': None, 'MICE': None, 'RF': None, 'KNN': None}
    df_orig_norm = pd.DataFrame(
        scaler_ref.transform(df_orig.drop('class', axis=1)),
        columns=df_orig.drop('class', axis=1).columns, index=df_orig.index)
    true_vals = df_orig_norm[target_col].values[mask_rows]
    feat_idx  = list(df_orig_norm.columns).index(target_col)
    df_feat   = df_sim.drop('class', axis=1)
    res       = {'n': len(mask_rows)}
    for method, imp in [
        ('MICE', IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)),
        ('RF',   IterativeImputer(estimator=RandomForestRegressor(random_state=0), max_iter=10, random_state=0)),
        ('KNN',  KNNImputer(n_neighbors=5)),
    ]:
        try:
            arr   = imp.fit_transform(df_feat)
            arr_n = scaler_ref.transform(arr)
            res[method] = round(float(np.sqrt(mean_squared_error(
                true_vals, arr_n[mask_rows, feat_idx]))), 4)
        except:
            res[method] = None
    try:
        X_n = pd.DataFrame(scaler_ref.transform(df_feat),
                           columns=df_feat.columns, index=df_sim.index)
        X_n['class'] = df_sim['class']
        df_f = chit_fast(X_n, col_method='mean', row_model='rf')
        res['CHIT'] = round(float(np.sqrt(mean_squared_error(
            true_vals, df_f[target_col].values[mask_rows]))), 4)
    except:
        res['CHIT'] = None
    return res

# ── 5-seed DL robustness ─────────────────────────────────────────────────────
def run_5seed_dl(df_sim, col_names):
    seeds = [0, 7, 21, 42, 99]
    results = []
    for seed in seeds:
        mice = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
        df_m = pd.DataFrame(mice.fit_transform(df_sim.copy()), columns=col_names)
        tr, te = train_test_split(df_m, test_size=0.2, random_state=seed)
        try:
            dl_m = run_dl(tr.iloc[:,:-1].values.astype(np.float32),
                          tr.iloc[:,-1].values.astype(np.int32),
                          te.iloc[:,:-1].values.astype(np.float32),
                          te.iloc[:,-1].values.astype(np.int32))
            mice_acc = dl_m['acc']
        except: mice_acc = 0.0
        df_norm = normalize_for_chit(df_sim)
        df_f = chit_fast(df_norm, col_method='mean', row_model='rf', seed=seed)
        tr, te = train_test_split(df_f, test_size=0.2, random_state=seed)
        try:
            dl_c = run_dl(tr.iloc[:,:-1].values.astype(np.float32),
                          tr.iloc[:,-1].values.astype(np.int32),
                          te.iloc[:,:-1].values.astype(np.float32),
                          te.iloc[:,-1].values.astype(np.int32))
            chit_acc = dl_c['acc']
        except: chit_acc = 0.0
        results.append({'seed': seed, 'CHIT': chit_acc, 'MICE': mice_acc})
        print(f"    seed={seed}: CHIT={chit_acc:.4f} MICE={mice_acc:.4f}")
    return results

# ── Ana deney fonksiyonu ─────────────────────────────────────────────────────
def run_experiment(df_sim, mech_label, df_orig, scaler_ref, target_col, mask_rows, run_seed=True):
    print(f"\n{'='*55}\nMEKANİZMA: {mech_label}\n{'='*55}")
    col_names   = df_sim.columns
    clf_names   = ['KNN','LR','SVC','DT','RFC','GNB','MLP','DL']
    num_classes = int(df_sim['class'].nunique())
    n_test      = int(len(df_sim) * 0.2)

    # RMSE
    print("RMSE hesaplanıyor...")
    rmse_res = compute_rmse(df_sim, target_col, mask_rows, df_orig, scaler_ref)
    print(f"  RMSE: {rmse_res}")

    # MICE (ham veri)
    print("MICE (ham)...", end='', flush=True); t0 = time.time()
    df_m = pd.DataFrame(IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                        random_state=0).fit_transform(df_sim.copy()), columns=col_names)
    tr_m, te_m = train_test_split(df_m, test_size=0.2, random_state=42)
    mice_res = run_classifiers(tr_m.iloc[:,:-1], tr_m.iloc[:,-1].astype(int),
                                te_m.iloc[:,:-1], te_m.iloc[:,-1].astype(int), num_classes)
    print(f" {time.time()-t0:.0f}s  mean={round(np.mean([v['acc'] for v in mice_res.values()]),4)}")
    del df_m, tr_m, te_m; gc.collect()

    # RF (ham veri)
    print("RF-Iter (ham)...", end='', flush=True); t0 = time.time()
    df_r = pd.DataFrame(IterativeImputer(estimator=RandomForestRegressor(), max_iter=10,
                        random_state=0).fit_transform(df_sim.copy()), columns=col_names)
    tr_r, te_r = train_test_split(df_r, test_size=0.2, random_state=42)
    rf_res = run_classifiers(tr_r.iloc[:,:-1], tr_r.iloc[:,-1].astype(int),
                              te_r.iloc[:,:-1], te_r.iloc[:,-1].astype(int), num_classes)
    print(f" {time.time()-t0:.0f}s  mean={round(np.mean([v['acc'] for v in rf_res.values()]),4)}")
    del df_r, tr_r, te_r; gc.collect()

    # KNN (ham veri)
    print("KNN-Imp (ham)...", end='', flush=True); t0 = time.time()
    df_k = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df_sim.copy()), columns=col_names)
    tr_k, te_k = train_test_split(df_k, test_size=0.2, random_state=42)
    knn_res = run_classifiers(tr_k.iloc[:,:-1], tr_k.iloc[:,-1].astype(int),
                               te_k.iloc[:,:-1], te_k.iloc[:,-1].astype(int), num_classes)
    print(f" {time.time()-t0:.0f}s  mean={round(np.mean([v['acc'] for v in knn_res.values()]),4)}")
    del df_k, tr_k, te_k; gc.collect()

    # 5-seed DL robustness
    seed_results = []
    if run_seed:
        print("5-seed DL robustness...")
        seed_results = run_5seed_dl(df_sim, col_names)

    # CHIT 42 konfigürasyon
    df_norm = normalize_for_chit(df_sim)
    chit_per_clf = {c: {'best_acc':0.0,'mean_acc':0.0,'std_acc':0.0,'all_accs':{},
                         'best_pred':[],'yte':[],'f1':0.0,'prec':0.0,'rec':0.0,
                         'val_acc_history':[],'train_acc_history':[]} for c in clf_names}
    all_accs   = {c: [] for c in clf_names}
    best_preds = {c: None for c in clf_names}
    best_ytes  = {c: None for c in clf_names}
    best_hist  = {c: None for c in clf_names}

    total = len(COL_METHODS) * len(ROW_MODELS); done = 0
    for cm_key, cm_name in COL_METHODS:
        for rm_key, rm_name in ROW_MODELS:
            done += 1
            key = f"{rm_name[:10]}_{cm_name[:8]}"
            t0  = time.time()
            print(f"  CHIT {done}/{total} ({cm_key},{rm_key})...", end='', flush=True)
            try:
                df_f = chit_fast(df_norm, col_method=cm_key, row_model=rm_key)
                tr, te = train_test_split(df_f, test_size=0.2, random_state=42)
                clf_res = run_classifiers(
                    tr.iloc[:,:-1], tr.iloc[:,-1].astype(int),
                    te.iloc[:,:-1], te.iloc[:,-1].astype(int), num_classes)
                for c in clf_names:
                    acc = clf_res[c]['acc']
                    all_accs[c].append(acc)
                    chit_per_clf[c]['all_accs'][key] = acc
                    if acc > chit_per_clf[c]['best_acc']:
                        chit_per_clf[c].update({
                            'best_acc': acc, 'f1':   clf_res[c]['f1'],
                            'prec':     clf_res[c]['prec'], 'rec': clf_res[c]['rec']})
                        best_preds[c] = np.array(clf_res[c]['pred'])
                        best_ytes[c]  = np.array(clf_res[c]['yte'])
                        if c == 'DL' and 'val_acc_history' in clf_res[c]:
                            best_hist[c] = {'val':   clf_res[c]['val_acc_history'],
                                            'train': clf_res[c]['train_acc_history']}
                print(f" {time.time()-t0:.0f}s best={max(clf_res[c]['acc'] for c in clf_names):.3f} ✓")
                del df_f, tr, te, clf_res; gc.collect()
            except Exception as e:
                print(f" HATA: {e}")
                for c in clf_names:
                    all_accs[c].append(0.0)
                    chit_per_clf[c]['all_accs'][key] = 0.0

    for c in clf_names:
        valid = [a for a in all_accs[c] if a > 0]
        chit_per_clf[c]['mean_acc']  = round(np.mean(valid), 4) if valid else 0.0
        chit_per_clf[c]['std_acc']   = round(np.std(valid),  4) if valid else 0.0
        chit_per_clf[c]['best_pred'] = best_preds[c].tolist() if best_preds[c] is not None else []
        chit_per_clf[c]['yte']       = best_ytes[c].tolist()  if best_ytes[c]  is not None else []
        if best_hist[c]:
            chit_per_clf[c]['val_acc_history']   = best_hist[c]['val']
            chit_per_clf[c]['train_acc_history'] = best_hist[c]['train']

    print("\nCHIT sonuçları:")
    for c in clf_names:
        v = chit_per_clf[c]
        print(f"  {c}: best={v['best_acc']} mean±std={v['mean_acc']}±{v['std_acc']}")

    # McNemar ve Wilson CI
    mcn_res = {}; ci_res = {}
    for c in clf_names:
        if best_preds[c] is not None and mice_res[c]['pred']:
            yt = np.array(chit_per_clf[c]['yte'])
            pc = np.array(chit_per_clf[c]['best_pred'])
            pm = np.array(mice_res[c]['pred'][:len(yt)])
            p, b, cv = mcnemar_test(yt, pc, pm)
        else:
            p, b, cv = 1.0, 0, 0
        mcn_res[c] = {'p': p, 'b': b, 'c': cv}
        ci_res[c]  = {'CHIT_ci': wci(chit_per_clf[c]['best_acc'], n=n_test),
                      'MICE_ci': wci(mice_res[c]['acc'], n=n_test)}
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"  McNemar {c}: p={p:.6f}({sig}) b={b} c={cv}")

    output = {
        'mech': mech_label, 'n_test': n_test, 'classifiers': clf_names,
        'pipeline': 'CHIT=normalized, MICE/RF/KNN=raw (original notebook)',
        'CHIT': chit_per_clf,
        'MICE': {c: {'acc':mice_res[c]['acc'],'f1':mice_res[c]['f1'],
                      'prec':mice_res[c]['prec'],'rec':mice_res[c]['rec'],
                      'pred':mice_res[c]['pred'],'yte':mice_res[c]['yte']}
                 for c in clf_names},
        'RF':  {c: {'acc':rf_res[c]['acc'],'f1':rf_res[c]['f1']} for c in clf_names},
        'KNN': {c: {'acc':knn_res[c]['acc'],'f1':knn_res[c]['f1']} for c in clf_names},
        'mcnemar': mcn_res, 'wilson_ci': ci_res,
        'rmse': rmse_res, 'seed_robustness_dl': seed_results,
    }

    fname = f'chit_complete_{mech_label.lower()}.json'
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    shutil.copy(fname, DRIVE_OUT + fname)

    cm = round(np.mean([chit_per_clf[c]['best_acc'] for c in clf_names]), 4)
    mm = round(np.mean([mice_res[c]['acc'] for c in clf_names]), 4)
    print(f"\n>>> {mech_label}: CHIT={cm}  MICE={mm}  gap={round(cm-mm,4):+}")
    print(f"    Kaydedildi: {DRIVE_OUT+fname}")
    return output

# ════════════════════════════════════════════════════════════════════════════════
# CKD
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*55 + "\nCKD VERİ SETİ\n" + "═"*55)
df = load_ckd()
print(f"CKD: {df.shape}, class: {df['class'].value_counts().to_dict()}")
scaler = StandardScaler(); scaler.fit(df.drop('class', axis=1))

# MCAR
df_mcar = sim_mcar(df, rate=0.1)
np.random.seed(42); tmp = df.copy()
mask = np.random.rand(*tmp.shape) < 0.1
mask[:, tmp.columns.get_loc('class')] = False; tmp[mask] = np.nan
mcar_rows = np.where(tmp['sc'].isnull() & df['sc'].notnull())[0]
res_mcar = run_experiment(df_mcar, 'MCAR', df, scaler, 'sc', mcar_rows)
del df_mcar; gc.collect()

# MAR
df_mar = df.copy()
mar_rows = np.where(df_mar['bp'].isnull())[0]
res_mar = run_experiment(df_mar, 'MAR', df, scaler, 'bp', mar_rows, run_seed=False)
del df_mar; gc.collect()

# MNAR
df_mnar = sim_mnar(df, col='hemo', thr=12)
mnar_rows = np.where(df_mnar['hemo'].isnull() & df['hemo'].notnull())[0]
res_mnar = run_experiment(df_mnar, 'MNAR', df, scaler, 'hemo', mnar_rows)
del df_mnar, df; gc.collect()

# ════════════════════════════════════════════════════════════════════════════════
# HDD
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*55 + "\nHDD VERİ SETİ\n" + "═"*55)
df_hdd = load_hdd()
print(f"HDD: {df_hdd.shape}, class: {df_hdd['class'].value_counts().to_dict()}")
scaler_hdd = StandardScaler(); scaler_hdd.fit(df_hdd.drop('class', axis=1))

# MCAR
df_hdd_mcar = sim_mcar(df_hdd, rate=0.1)
np.random.seed(42); tmp = df_hdd.copy()
mask = np.random.rand(*tmp.shape) < 0.1
mask[:, tmp.columns.get_loc('class')] = False; tmp[mask] = np.nan
hdd_mcar_rows = np.where(tmp['trestbps'].isnull() & df_hdd['trestbps'].notnull())[0]
res_hdd_mcar = run_experiment(df_hdd_mcar, 'HDD_MCAR', df_hdd, scaler_hdd, 'trestbps', hdd_mcar_rows)
del df_hdd_mcar; gc.collect()

# MAR — chol yüksek olanlarda trestbps eksik
df_hdd_mar = df_hdd.copy()
rng = np.random.RandomState(42)
high_chol = df_hdd['chol'] > df_hdd['chol'].median()
mar_idx = df_hdd.index[high_chol & (rng.rand(len(df_hdd)) < 0.2)]
df_hdd_mar.loc[mar_idx, 'trestbps'] = np.nan
hdd_mar_rows = np.where(df_hdd_mar['trestbps'].isnull() & df_hdd['trestbps'].notnull())[0]
res_hdd_mar = run_experiment(df_hdd_mar, 'HDD_MAR', df_hdd, scaler_hdd, 'trestbps', hdd_mar_rows, run_seed=False)
del df_hdd_mar; gc.collect()

# MNAR — chol > 250
df_hdd_mnar = sim_mnar(df_hdd, col='chol', thr=250)
hdd_mnar_rows = np.where(df_hdd_mnar['chol'].isnull() & df_hdd['chol'].notnull())[0]
res_hdd_mnar = run_experiment(df_hdd_mnar, 'HDD_MNAR', df_hdd, scaler_hdd, 'chol', hdd_mnar_rows)
del df_hdd_mnar, df_hdd; gc.collect()

# ════════════════════════════════════════════════════════════════════════════════
# MPED
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*55 + "\nMPED VERİ SETİ\n" + "═"*55)
df_mped = load_mped()
print(f"MPED: {df_mped.shape}, class: {df_mped['class'].value_counts().to_dict()}")
scaler_mped = StandardScaler(); scaler_mped.fit(df_mped.drop('class', axis=1))

# MCAR
df_mped_mcar = sim_mcar(df_mped, rate=0.1)
np.random.seed(42); tmp = df_mped.copy()
mask = np.random.rand(*tmp.shape) < 0.1
mask[:, tmp.columns.get_loc('class')] = False; tmp[mask] = np.nan
mped_mcar_rows = np.where(tmp['BDNF_N'].isnull() & df_mped['BDNF_N'].notnull())[0]
res_mped_mcar = run_experiment(df_mped_mcar, 'MPED_MCAR', df_mped, scaler_mped, 'BDNF_N', mped_mcar_rows, run_seed=False)
del df_mped_mcar; gc.collect()

# MAR — doğal eksik + simüle
df_mped_mar = df_mped.copy()
mar_nat = np.where(df_mped_mar['BDNF_N'].isnull())[0]
if len(mar_nat) > 5:
    mped_mar_rows = mar_nat
else:
    rng2 = np.random.RandomState(42)
    df_mped_mar.loc[rng2.rand(len(df_mped)) < 0.1, 'BDNF_N'] = np.nan
    mped_mar_rows = np.where(df_mped_mar['BDNF_N'].isnull() & df_mped['BDNF_N'].notnull())[0]
res_mped_mar = run_experiment(df_mped_mar, 'MPED_MAR', df_mped, scaler_mped, 'BDNF_N', mped_mar_rows, run_seed=False)
del df_mped_mar; gc.collect()

# MNAR — alt çeyrek altı
df_mped_mnar = sim_mnar(df_mped, col='BDNF_N', quantile=0.25)
mped_mnar_rows = np.where(df_mped_mnar['BDNF_N'].isnull() & df_mped['BDNF_N'].notnull())[0]
res_mped_mnar = run_experiment(df_mped_mnar, 'MPED_MNAR', df_mped, scaler_mped, 'BDNF_N', mped_mnar_rows, run_seed=False)
del df_mped_mnar, df_mped; gc.collect()

# ════════════════════════════════════════════════════════════════════════════════
# ÖZET
# ════════════════════════════════════════════════════════════════════════════════
print("\n\n" + "═"*55)
print("TÜM 9 DENEY TAMAMLANDI")
print("═"*55)
clf_names = ['KNN','LR','SVC','DT','RFC','GNB','MLP','DL']
all_results = [
    ('CKD_MCAR',  res_mcar),    ('CKD_MAR',   res_mar),    ('CKD_MNAR',  res_mnar),
    ('HDD_MCAR',  res_hdd_mcar),('HDD_MAR',   res_hdd_mar),('HDD_MNAR',  res_hdd_mnar),
    ('MPED_MCAR', res_mped_mcar),('MPED_MAR',  res_mped_mar),('MPED_MNAR', res_mped_mnar),
]
for mech, res in all_results:
    cm = round(np.mean([res['CHIT'][c]['best_acc'] for c in clf_names]), 4)
    mm = round(np.mean([res['MICE'][c]['acc']      for c in clf_names]), 4)
    print(f"  {mech:<12}: CHIT={cm:.4f}  MICE={mm:.4f}  gap={round(cm-mm,4):+.4f}")

# ── Otomatik indir ────────────────────────────────────────────────────────────
print("\nDosyalar indiriliyor...")
json_files = [
    'chit_complete_mcar.json',     'chit_complete_mar.json',     'chit_complete_mnar.json',
    'chit_complete_hdd_mcar.json', 'chit_complete_hdd_mar.json', 'chit_complete_hdd_mnar.json',
    'chit_complete_mped_mcar.json','chit_complete_mped_mar.json','chit_complete_mped_mnar.json',
]
for fname in json_files:
    try:
        files.download(fname)
        print(f"  {fname} ✓")
    except:
        print(f"  {fname} Drive'da: {DRIVE_OUT+fname}")
