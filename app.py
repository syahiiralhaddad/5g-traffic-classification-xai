"""
QoSight Backend — Flask API with Real-Time SSE Progress
=======================================================
Usage:
  pip install -r requirements.txt
  export HF_TOKEN=hf_xxx        # optional, enables LLM-based XAI (Windows: set HF_TOKEN=hf_xxx)
  python app.py
  Open: http://localhost:5000
"""

import os, sys, json, time, uuid, warnings, traceback, threading, queue
from copy import deepcopy

import numpy as np
import pandas as pd
import requests as req_lib
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# HuggingFace token is read from the environment — never hard-code secrets
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
LLM_MODEL = "Qwen/Qwen3.5-397B-A17B"
RUN_XAI   = os.environ.get("RUN_XAI", "1") == "1"

DEFAULT_DATA_PATH = os.path.join("data", "5g_traffic_dataset.csv")
FEATURES = ['packet_loss', 'throughput', 'latency', 'jitter', 'packet_length']
TARGET   = 'traffic_type'
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
progress_queues = {}
result_store    = {}

def push(q, step, pct, message, data=None):
    p = {'step': step, 'pct': pct, 'message': message}
    if data: p['data'] = data
    q.put(p)

class XAIExplainer:
    """XAI Module — prompt identik dengan ML asli (JANGAN DIUBAH)"""

    def __init__(self, hf_token, model_id):
        self.hf_token  = hf_token
        self.model_id  = model_id
        self.api_url   = "https://router.huggingface.co/v1/chat/completions"
        self.headers   = {"Authorization": f"Bearer {hf_token}",
                          "Content-Type": "application/json"}
        self.explanations = {}

    def _strip_thinking(self, text):
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _call_llm(self, prompt, max_tokens=8000, max_retries=5):
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }

        for attempt in range(1, max_retries + 1):
            # Buat session baru tiap attempt — hindari stale connection (fix ConnectionResetError 10054)
            session = req_lib.Session()
            retry_cfg = Retry(total=0, raise_on_status=False)
            adapter = HTTPAdapter(max_retries=retry_cfg, pool_connections=1, pool_maxsize=1)
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            try:
                response = session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=(30, 300)  # (connect timeout, read timeout)
                )
                if response.status_code == 200:
                    raw_text = response.json()['choices'][0]['message']['content']
                    stripped = self._strip_thinking(raw_text)
                    print(f"  [DEBUG] raw_len={len(raw_text)} stripped_len={len(stripped)} preview={stripped[:80]!r}")
                    return stripped
                elif response.status_code == 503:
                    print(f"  Model loading... retry {attempt}/{max_retries} (wait 30s)")
                    time.sleep(30)
                elif response.status_code == 429:
                    print(f"  Rate limit, retry {attempt}/{max_retries} (wait 60s)")
                    time.sleep(60)
                else:
                    return f"[LLM Error {response.status_code}] {response.text[:200]}"
            except req_lib.exceptions.Timeout:
                print(f"  Timeout, retry {attempt}/{max_retries} (wait 15s)...")
                time.sleep(15)
            except (req_lib.exceptions.ConnectionError, ConnectionResetError, OSError) as e:
                wait = 20 + attempt * 10  # makin lama makin nunggu: 30s, 40s, 50s...
                print(f"  Connection reset (attempt {attempt}/{max_retries}), wait {wait}s... [{e}]")
                time.sleep(wait)
            except Exception as e:
                print(f"  Unexpected error, retry {attempt}/{max_retries} (wait 10s)... [{e}]")
                time.sleep(10)
            finally:
                session.close()  # pastikan koneksi ditutup tiap attempt

        return "[LLM Error] Max retries exceeded after 5 attempts."

    def _extract_distinguishing(self, traffic_type, qos_stats):
        distinguishing = []
        if qos_stats is None:
            return distinguishing
        for feature in qos_stats['feature'].unique():
            class_row = qos_stats[
                (qos_stats['traffic_type'] == traffic_type) &
                (qos_stats['feature'] == feature)
            ]['mean'].values
            if len(class_row) == 0:
                continue
            class_val  = float(class_row[0])
            if np.isnan(class_val): continue
            other_mean = float(qos_stats[
                (qos_stats['traffic_type'] != traffic_type) &
                (qos_stats['feature'] == feature)
            ]['mean'].mean())
            if np.isnan(other_mean): continue
            diff_pct  = abs((class_val - other_mean) / (other_mean + 1e-10)) * 100
            if diff_pct > 50:
                direction = "higher" if class_val > other_mean else "lower"
                distinguishing.append(
                    f"{feature}: {diff_pct:.0f}% {direction} than other classes "
                    f"(mean: {class_val:.2f})"
                )
        return distinguishing

    def _build_classification_prompt(self, traffic_type, feature_importance,
                                     class_profile, distinguishing_features):
        # Full feature importance
        fi_text = "".join(
            f"  - {row['feature']}: {row['importance_mean']*100:.1f}%\n"
            for _, row in feature_importance.iterrows()
        )
        # Parse nama fitur yang masuk distinguishing features
        dist_feat_names = set()
        for d in distinguishing_features:
            feat = d.split(":")[0].strip()
            dist_feat_names.add(feat)
        # QoS profile target class — HANYA fitur yang distinguishing
        profile = class_profile.get(traffic_type, {})
        profile_text = "".join(
            f"  - {feat}: mean={float(stats['mean']):.2f}\n"
            for feat, stats in profile.items()
            if feat in dist_feat_names
        )
        # QoS profile kelas lain — HANYA fitur yang distinguishing
        other_classes_text = ""
        for other_type, other_profile in class_profile.items():
            if other_type == traffic_type:
                continue
            other_classes_text += f"\n  {other_type}:\n"
            for feat, stats in other_profile.items():
                if feat in dist_feat_names:
                    other_classes_text += f"    - {feat}: mean={float(stats['mean']):.2f}\n"
        # Distinguishing features text
        dist_text = (
            "\n".join(f"  - {d}" for d in distinguishing_features)
            if distinguishing_features
            else "  - No significant distinguishing features found."
        )

        # ── PROMPT IDENTIK DENGAN ML ASLI ──────────────────────────────────
        prompt = (
            f"You are a network traffic classification expert analyzing QoS (Quality of Service) "
            f"parameters from a 5G network.\n\n"

            f"STRICT RULES — MUST FOLLOW ALL:\n"
            f"1. Use ONLY the QoS parameters listed under KEY DISTINGUISHING FEATURES below. "
            f"Do NOT mention any other QoS parameter even if you think it is relevant.\n"
            f"2. Do NOT mention real-world applications, services, or use cases "
            f"(e.g. do NOT say 'file transfer', 'streaming', 'video conferencing', 'gaming', etc.).\n"
            f"3. Always cite exact mean values and importance % when referencing a feature.\n"
            f"4. Respect direction strictly: if a feature is stated as HIGHER, never imply it is lower "
            f"(and vice versa).\n"
            f"5. Write exactly 3-4 sentences. No bullet points.\n\n"

            f"DATA:\n"
            f"Traffic class being explained: {traffic_type}\n\n"

            f"Feature importance (contribution of each QoS parameter to model classification):\n"
            f"{fi_text}\n"

            f"QoS mean values of {traffic_type} — for the distinguishing features only:\n"
            f"{profile_text}\n"

            f"QoS mean values of other classes — for the distinguishing features only:\n"
            f"{other_classes_text}\n"

            f"KEY DISTINGUISHING FEATURES of {traffic_type} vs all other classes\n"
            f"(you are ONLY allowed to reference these features in your explanation):\n"
            f"{dist_text}\n\n"

            f"TASK: In 3-4 sentences, explain WHY this traffic is classified as \"{traffic_type}\". "
            f"Your explanation must cover:\n"
            f"  (a) Which distinguishing QoS parameter values set this class apart — cite their exact means\n"
            f"  (b) Why those values — given their feature importance % — cause the model to classify "
            f"this traffic as {traffic_type}\n"
            f"  (c) What the combination of these QoS values indicates about the traffic behavior, "
            f"based strictly on the numbers above — no assumptions, no application names\n\n"

            f"Explanation:"
        )
        return prompt

    def explain_all_classes(self, feature_importance, class_profiles, qos_stats, progress_q=None):
        classes = sorted(class_profiles.keys())
        # Pre-fill supaya frontend tidak pernah dapat undefined
        for tt in classes:
            self.explanations[tt] = ""

        for i, traffic_type in enumerate(classes):
            if progress_q:
                pct = 88 + int(i / len(classes) * 10)
                push(progress_q, 4, pct, f"Generating XAI for {traffic_type}...")

            if i > 0:
                print(f"  Waiting 10s before next class...")
                time.sleep(10)

            print(f"\n→ Generating explanation for: {traffic_type.upper()}...")
            try:
                distinguishing = self._extract_distinguishing(traffic_type, qos_stats)
                prompt = self._build_classification_prompt(
                    traffic_type, feature_importance, class_profiles, distinguishing
                )
                explanation = self._call_llm(prompt)
                if not explanation or len(explanation.strip()) < 50:
                    print(f"  Short/empty response (len={len(explanation.strip() if explanation else '')}), retrying with more tokens...")
                    time.sleep(10)
                    explanation = self._call_llm(prompt, max_tokens=12000)
                if not explanation or len(explanation.strip()) < 20:
                    explanation = f"[XAI generation failed for {traffic_type} — try running again]"
                self.explanations[traffic_type] = explanation
                print(f"  Done: {traffic_type} (len={len(explanation)})")
            except Exception as e:
                print(f"  Error for {traffic_type}: {e}")
                self.explanations[traffic_type] = f"[XAI error: {str(e)[:120]}]"

        return self.explanations

def run_pipeline(filepath, job_id):
    q = progress_queues[job_id]
    try:
        push(q, 1, 5, "Loading and validating data...")
        data = pd.read_excel(filepath) if filepath.endswith('.xlsx') else pd.read_csv(filepath)
        missing = [c for c in FEATURES+[TARGET] if c not in data.columns]
        if missing: raise ValueError(f"Missing columns: {missing}")

        if 'date' in data.columns and 'time' in data.columns:
            dt_str = data['date'].astype(str)+' '+data['time'].astype(str)
            data['datetime'] = pd.to_datetime(dt_str, dayfirst=True, errors='coerce')
        elif 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        else:
            data['datetime'] = pd.NaT

        data = data.dropna(subset=FEATURES+[TARGET]).reset_index(drop=True)
        if data['datetime'].notna().any():
            data = data.sort_values('datetime').reset_index(drop=True)

        push(q, 1, 12, f"Loaded {len(data):,} records — {data[TARGET].nunique()} classes")

        push(q, 2, 18, "Encoding labels and splitting data (80/20 stratified)...")
        le = LabelEncoder()
        X  = data[FEATURES].copy(); y = le.fit_transform(data[TARGET])
        classes = [str(c) for c in le.classes_]
        cw = 'balanced' if np.bincount(y).min()/len(y)*100 < 15 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        train_idx = X_train.index.tolist(); test_idx = X_test.index.tolist()

        push(q, 2, 24, "Scaling features with StandardScaler...")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train); X_te = scaler.transform(X_test)

        push(q, 3, 30, "Initializing base learners: Random Forest, SVM, KNN...")
        base_configs = [
            ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=20,
                min_samples_split=5, max_features='sqrt', class_weight=cw, random_state=42, n_jobs=-1)),
            ('SVM', SVC(C=10.0, gamma='scale', kernel='rbf', probability=True, class_weight=cw, random_state=42)),
            ('KNN', KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean', n_jobs=-1))
        ]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        n_cls = len(classes)
        oof = np.zeros((len(X_tr), len(base_configs), n_cls))
        fold_models = []; fi_hist = []
        # Kumpulkan CV accuracy per base model per fold (sesuai tabel: base model = validation fold)
        fold_cv_scores = {nm: [] for nm, _ in base_configs}

        for fi, (tr_i, val_i) in enumerate(skf.split(X_tr, y_train), 1):
            push(q, 3, 30+fi*7, f"Fold {fi}/5 — training KNN, SVM, Random Forest on {len(tr_i):,} samples...")
            fd = {}
            Xft, Xfv = X_tr[tr_i], X_tr[val_i]; yft = y_train[tr_i]
            for mi, (nm, tmpl) in enumerate(base_configs):
                m = deepcopy(tmpl); m.fit(Xft, yft)
                if nm == 'SVM':
                    m = CalibratedClassifierCV(m, cv=3, method='sigmoid'); m.fit(Xft, yft)
                oof[val_i, mi, :] = m.predict_proba(Xfv)
                # Catat validation accuracy tiap fold tiap model
                fold_cv_scores[nm].append(float(accuracy_score(y_train[val_i], m.predict(Xfv))))
                fd[nm] = m
                if nm == 'Random Forest' and hasattr(m, 'feature_importances_'):
                    fi_hist.append(m.feature_importances_.copy())
            fold_models.append(fd)

        push(q, 3, 68, "Training meta-learner (Logistic Regression) on OOF predictions...")
        meta_tr = oof.reshape(len(X_tr), -1)
        meta_m  = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42, n_jobs=-1)
        meta_m.fit(meta_tr, y_train)
        cv_sc = []
        for tr_i, val_i in skf.split(meta_tr, y_train):
            tmp = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42, n_jobs=-1)
            tmp.fit(meta_tr[tr_i], y_train[tr_i])
            cv_sc.append(accuracy_score(y_train[val_i], tmp.predict(meta_tr[val_i])))
        cv_sc = np.array(cv_sc)

        push(q, 3, 76, "Predicting test set and computing metrics...")
        all_te = np.zeros((5, len(X_te), len(base_configs), n_cls))
        for fi, fd in enumerate(fold_models):
            for mi, (nm, _) in enumerate(base_configs):
                all_te[fi,:,mi,:] = fd[nm].predict_proba(X_te)
        meta_te  = all_te.mean(axis=0).reshape(len(X_te), -1)
        y_te_p   = meta_m.predict(meta_te); y_te_pr  = meta_m.predict_proba(meta_te)
        y_tr_p   = meta_m.predict(meta_tr); y_tr_pr  = meta_m.predict_proba(meta_tr)

        acc     = float(accuracy_score(y_test, y_te_p))
        bal_acc = float(balanced_accuracy_score(y_test, y_te_p))
        f1      = float(f1_score(y_test, y_te_p, average='weighted', zero_division=0))
        pr      = float(precision_score(y_test, y_te_p, average='weighted', zero_division=0))
        rc      = float(recall_score(y_test, y_te_p, average='weighted', zero_division=0))
        cm      = confusion_matrix(y_test, y_te_p)

        # Base model accuracy = rata-rata CV validation accuracy per fold (sebelum meta-learner)
        base_model_acc = {nm: float(np.mean(scores)) for nm, scores in fold_cv_scores.items()}

        push(q, 3, 82, f"Accuracy: {float(acc)*100:.1f}%  |  F1: {float(f1)*100:.1f}%  |  CV: {float(cv_sc.mean())*100:.1f}%")

        push(q, 3, 84, "Building full predictions dataset...")
        fp = np.zeros(len(data), dtype=int); fpr = np.zeros((len(data), n_cls))
        for i, idx in enumerate(train_idx): fp[idx]=y_tr_p[i]; fpr[idx]=y_tr_pr[i]
        for i, idx in enumerate(test_idx):  fp[idx]=y_te_p[i]; fpr[idx]=y_te_pr[i]
        pred_lbl = [str(x) for x in le.inverse_transform(fp)]; conf = fpr.max(axis=1)
        correct  = [p == t for p, t in zip(pred_lbl, data[TARGET].values)]

        push(q, 3, 86, "Calculating feature importance (Random Forest averaged across folds)...")
        avg_fi = np.mean(fi_hist, axis=0) if fi_hist else np.ones(len(FEATURES))/len(FEATURES)
        fi_df  = pd.DataFrame({'feature': FEATURES, 'importance_mean': avg_fi})\
                   .sort_values('importance_mean', ascending=False).reset_index(drop=True)

        push(q, 3, 87, "Computing per-class QoS profiles...")
        results_df = data.copy(); results_df['predicted_traffic']=pred_lbl; results_df['confidence']=conf
        qos_rows = []
        for tt in sorted(set(pred_lbl)):
            sub = results_df[results_df['predicted_traffic']==tt]
            for feat in FEATURES:
                qos_rows.append({'traffic_type':tt,'feature':feat,'mean':float(sub[feat].mean()) if not sub[feat].empty else 0.0,
                                 'std':float(sub[feat].std()),'min':float(sub[feat].min()),'max':float(sub[feat].max())})
        qos_df = pd.DataFrame(qos_rows)
        profiles = {}
        for tt in sorted(set(pred_lbl)):
            s = qos_df[qos_df['traffic_type']==tt]
            profiles[tt] = {r['feature']:{'mean':float(r['mean']),'range':f"[{float(r['min']):.2f}-{float(r['max']):.2f}]"} for _,r in s.iterrows()}

        push(q, 3, 88, "Running time series analysis...")
        tsa = {}
        if results_df['datetime'].notna().any():
            df_t = results_df.dropna(subset=['datetime']).copy()
            df_t['hour'] = df_t['datetime'].dt.hour; df_t['date_only'] = df_t['datetime'].dt.date
            hrs = [int(h) for h in sorted(df_t['hour'].unique())]; dates = sorted(df_t['date_only'].unique()); nd = len(dates)
            tts = [str(t) for t in sorted(df_t['predicted_traffic'].unique())]
            hp = df_t.groupby(['hour','predicted_traffic']).size().unstack(fill_value=0)
            hourly = {str(tt):[round(float(hp.loc[h,tt])/max(nd,1),2) if h in hp.index and tt in hp.columns else 0.0 for h in hrs] for tt in tts}
            dc = {str(tt):[int((df_t[df_t['predicted_traffic']==tt]['date_only']==d).sum()) for d in dates] for tt in tts}
            ld = dates[-1]; last_df = df_t[df_t['date_only']==ld]
            lp = last_df.groupby(['hour','predicted_traffic']).size().unstack(fill_value=0)
            dh = {str(tt):[int(lp.loc[h,tt]) if h in lp.index and tt in lp.columns else 0 for h in hrs] for tt in tts}
            tsa = {'hours':hrs,'hourly':hourly,'daily_dates':[str(d) for d in dates],'daily_counts':dc,'daily_hourly':dh}

        xai_exp = {}
        if RUN_XAI and HF_TOKEN:
            push(q, 4, 88, "Calling LLM to generate XAI explanations...")
            try:
                xai_exp = XAIExplainer(HF_TOKEN, LLM_MODEL).explain_all_classes(fi_df, profiles, qos_df, progress_q=q)
            except Exception as e:
                print(f"XAI error: {e}"); xai_exp = {tt: f"XAI error: {e}" for tt in classes}
        else:
            push(q, 4, 90, "XAI skipped.")

        push(q, 4, 98, "Finalizing results...")
        preds_list = [{
            'datetime':          str(results_df['datetime'].iloc[i]),
            'actual_traffic':    str(data[TARGET].iloc[i]),
            'predicted_traffic': str(pred_lbl[i]),
            'confidence':        round(float(conf[i]), 6),
            'correct':           bool(correct[i]),
            'packet_loss':       float(data['packet_loss'].iloc[i]),
            'throughput':        float(data['throughput'].iloc[i]),
            'latency':           float(data['latency'].iloc[i]),
            'jitter':            float(data['jitter'].iloc[i]),
            'packet_length':     float(data['packet_length'].iloc[i]),
        } for i in range(len(data))]

        result = {'success':True,'total':int(len(data)),'classes':classes,
                  'metrics':{
                      'accuracy':               acc,
                      'balanced_accuracy':       bal_acc,
                      'f1':                     f1,
                      'precision':              pr,
                      'recall':                 rc,
                      'cv_mean':                float(cv_sc.mean()),
                      'cv_std':                 float(cv_sc.std()),
                      'confusion_matrix':       cm.tolist(),
                      'confusion_matrix_labels': classes,
                      'base_models':            base_model_acc,
                  },
                  'feature_importance': [{'feature': str(r['feature']), 'importance': float(r['importance_mean'])} for _,r in fi_df.iterrows()],
                  'predictions': preds_list, 'tsa': tsa, 'xai_explanations': xai_exp}

        result_store[job_id] = result
        push(q, 4, 100, "Classification complete!", data=result)

    except Exception as e:
        traceback.print_exc()
        err = {'success':False,'error':str(e)}
        result_store[job_id] = err
        q.put({'step':0,'pct':0,'message':f'ERROR: {e}','data':err})
    finally:
        q.put(None)

@app.route('/')
def index(): return send_from_directory('.', 'qosight_home.html')

@app.route('/health')
def health(): return jsonify({'status':'ok'})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files: return jsonify({'success':False,'error':'No file'}),400
    f = request.files['file']
    if not f.filename.endswith('.csv'): return jsonify({'success':False,'error':'CSV only'}),400
    fname = f"{uuid.uuid4().hex}_{f.filename}"; path = os.path.join(UPLOAD_FOLDER, fname); f.save(path)
    try:
        df = pd.read_csv(path); missing = [c for c in FEATURES+[TARGET] if c not in df.columns]
        if missing: os.remove(path); return jsonify({'success':False,'error':f"Missing: {missing}"}),400
        return jsonify({'success':True,'filename':fname,'rows':len(df)})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

@app.route('/classify/start', methods=['POST'])
def classify_start():
    body = request.json or {}; source = body.get('source','sample')
    if source == 'upload':
        fname = body.get('filename',''); fp = os.path.join(UPLOAD_FOLDER, fname)
        if not fname or not os.path.exists(fp): return jsonify({'success':False,'error':'File not found'}),400
    else:
        fp = DEFAULT_DATA_PATH
        if not os.path.exists(fp):
            return jsonify({'success':False,'error':f'Dataset not found: "{DEFAULT_DATA_PATH}". Put it in the data/ folder.'}),400
    job_id = uuid.uuid4().hex
    progress_queues[job_id] = queue.Queue()
    threading.Thread(target=run_pipeline, args=(fp, job_id), daemon=True).start()
    return jsonify({'success':True,'job_id':job_id})

@app.route('/classify/progress/<job_id>')
def classify_progress(job_id):
    if job_id not in progress_queues: return jsonify({'error':'Job not found'}),404
    def generate():
        q = progress_queues[job_id]
        while True:
            msg = q.get()
            if msg is None: yield 'data: {"done":true}\n\n'; break
            yield f"data: {json.dumps(msg)}\n\n"
        progress_queues.pop(job_id, None)
    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

if __name__ == '__main__':
    print("\n"+"="*55)
    print("  QoSight Backend — Flask + SSE Real-Time Progress")
    print("="*55)
    print(f"  Dataset : {DEFAULT_DATA_PATH}")
    print(f"  XAI     : {'ON — '+LLM_MODEL if RUN_XAI else 'OFF'}")
    print(f"\n  ✅  Open: http://localhost:5000\n"+"="*55+"\n")
    app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)