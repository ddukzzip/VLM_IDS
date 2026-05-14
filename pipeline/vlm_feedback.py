"""
VLM 피드백 파이프라인 v2
순서: CT filter → density image → VLM 분석 → ACD(VLM boost) → RF
"""

import sys, os, re, json
import numpy as np
from PIL import Image
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

sys.path.insert(0, "/home/oper/LabelDiffusion")
from models.density_map      import DensityMapBuilder
from models.density_detector import DensityNoisyDetector
from models.baselines         import CoTeaching
from models.augmentation      import DataAugmentor
from models.adaptive_cluster_density import AdaptiveClusterDensity

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
CKPT     = ("/home/oper/VLM_IDS/results/"
            "checkpoints_lora_v4/best_lora")
IMG_TMP  = "/tmp/vlm_density_tmp.png"

SYSTEM = (
    "당신은 네트워크 트래픽 density map을 분석하는 "
    "IDS 전문가입니다. "
    "입력 이미지(224x224 RGB)는 3채널로 구성됩니다: "
    "R채널=benign 트래픽 밀도(밝을수록 높음), "
    "G채널=malicious 트래픽 밀도(밝을수록 높음), "
    "B채널=Shannon Entropy Map "
    "(R과 G가 동시에 존재할수록 밝음=노이즈 위험 구간). "
    "x축=bin(0~49, 값의 크기 구간), "
    "y축=feature(0~49, 패킷 순서). "
    "B채널이 밝게 빛나는 구간=be/ma 혼재=noisy 집중. "
    "반드시 JSON 형식으로만 답하시오.")


class VLMFeedbackPipeline:

    def __init__(self, load_vlm=True):
        self.model     = None
        self.processor = None
        if load_vlm:
            self._load_vlm()

    def _load_vlm(self):
        print("VLM 로드 중...")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base, CKPT)
        self.model.eval()
        print("VLM 로드 완료")

    def _build_density_image(self, X, y, n_bins=50):
        """
        3채널 density image 생성 (224x224)
        CH1(R): benign density (log 강조)
        CH2(G): malicious density (log 강조)
        CH3(B): Shannon Entropy Map
                - be/ma 혼재 구간 밝게
                - density 가중치 적용
                - 노이즈 집중 구간이 밝게 빛남
        """
        db = DensityMapBuilder(
            n_bins=n_bins, cons_thresh=0.7,
            k=min(20, len(X)//4))
        db.fit(X, y)
        dimg   = db.build()
        be_map = dimg[:, :, 0]
        ma_map = dimg[:, :, 1]

        # CH1, CH2: log 스케일 강조
        be_log = np.log1p(be_map * 10)
        ma_log = np.log1p(ma_map * 10)
        be_norm = np.clip(
            be_log / (be_log.max() + 1e-8), 0, 1)
        ma_norm = np.clip(
            ma_log / (ma_log.max() + 1e-8), 0, 1)

        # CH3: Shannon Entropy Map
        total = be_map + ma_map + 1e-8
        p1 = be_map / total
        p2 = ma_map / total
        entropy = -(
            p1 * np.log2(p1 + 1e-9) +
            p2 * np.log2(p2 + 1e-9))
        # density 가중치: 데이터 없는 빈 공간 제외
        density_mask = np.tanh(total * 5)
        conflict = entropy * density_mask
        conflict_norm = np.clip(
            conflict / (conflict.max() + 1e-8), 0, 1)

        def to_224(arr, gamma=1.2):
            arr = np.power(
                np.clip(arr, 0, 1), 1.0/gamma)
            img = Image.fromarray(
                (arr * 255).astype(np.uint8))
            return np.array(img.resize(
                (224, 224), Image.LANCZOS))

        combined = np.stack([
            to_224(be_norm),
            to_224(ma_norm),
            to_224(conflict_norm)
        ], axis=-1).astype(np.uint8)

        return combined, be_map, ma_map, conflict, db

    def _vlm_analyze(self, img_arr, dataset="unknown",
                     noise_rate=0.45):
        """
        VLM으로 density image 분석
        CH1(R), CH2(G), CH3(B) 전부 분석
        → risky_features, risky_bins 파싱
        """
        if self.model is None:
            return {"risky_features": [],
                    "risky_bins": [],
                    "explanation": "VLM 미사용"}

        img = Image.fromarray(img_arr).convert("RGB")
        question = (
            "Analyze this density map.\n"
            "B channel (blue) bright areas = "
            "noisy label risk zones.\n"
            "Find feature/bin indices where "
            "B channel is brightest.\n"
            "Return ONLY valid JSON. "
            "No explanation, no markdown.\n"
            "Example: "
            "{\"risky_features\": [3, 7, 9], "
            "\"risky_bins\": [0, 1, 2], "
            "\"reason\": \"B channel bright\"}\n"
            "Your response (JSON only):")

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question}]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True)
        image_inputs, _ = process_vision_info([messages])
        inputs = self.processor(
            text=[text], images=image_inputs,
            return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=300,
                do_sample=False)
        explanation = self.processor.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)

        risky_features = []
        risky_bins     = []
        ch1_features   = []
        ch2_features   = []
        parse_success  = False

        # 0순위: RESULT: 뒤 JSON 파싱 (학습 데이터 형식)
        m_result = re.search(
            r'RESULT:\s*(\{.*?\})',
            explanation, re.DOTALL)
        if m_result:
            try:
                data = json.loads(m_result.group(1))
                risky_features = [
                    int(x) for x in
                    data.get("risky_features",[])]
                risky_bins = [
                    int(x) for x in
                    data.get("risky_bins",[])]
                if risky_features:
                    parse_success = True
            except:
                pass

        # 1순위: JSON 파싱 (여러 패턴 시도)
        if not parse_success:
            json_patterns = [
                r'\{[^{}]*"risky_features"[^{}]*\}',
                r'\{.*?"risky_features".*?\}',
            ]
            for pat in json_patterns:
                try:
                    m = re.search(pat, explanation,
                                  re.DOTALL)
                    if m:
                        txt = m.group().replace(
                            "'", "\"")
                        data = json.loads(txt)
                        risky_features = [
                            int(x) for x in
                            data.get("risky_features",[])]
                        risky_bins = [
                            int(x) for x in
                            data.get("risky_bins",[])]
                        if risky_features:
                            parse_success = True
                            break
                except:
                    continue

        # 2순위: 숫자 리스트 패턴
        if not parse_success:
            # "risky_features" 키워드 뒤 숫자 추출
            m_rf = re.search(
                r'risky_features[^\[]*\[([\d,\s]+)\]',
                explanation, re.IGNORECASE)
            if m_rf:
                try:
                    risky_features = [
                        int(x.strip()) for x in
                        m_rf.group(1).split(",")
                        if x.strip().isdigit()]
                    parse_success = bool(risky_features)
                except: pass

            m_rb = re.search(
                r'risky_bins[^\[]*\[([\d,\s]+)\]',
                explanation, re.IGNORECASE)
            if m_rb:
                try:
                    risky_bins = [
                        int(x.strip()) for x in
                        m_rb.group(1).split(",")
                        if x.strip().isdigit()]
                except: pass

        # 3순위: 모든 숫자 리스트 추출
        if not parse_success:
            all_lists = re.findall(
                r'\[([\d,\s]+)\]', explanation)
            parsed = []
            for lst in all_lists:
                nums = [int(x.strip())
                        for x in lst.split(",")
                        if x.strip().isdigit()
                        and int(x.strip()) < 50]
                if 1 <= len(nums) <= 10:
                    parsed.append(nums)
            if len(parsed) >= 1:
                risky_features = parsed[0]
            if len(parsed) >= 2:
                risky_bins = parsed[1]

        print("  [VLM] parsed={} features={} bins={}".format(
            parse_success, risky_features, risky_bins))
        print("  [VLM raw] {}".format(
            explanation[:150].replace("\n"," ")))

        return {
            "risky_features": risky_features,
            "risky_bins":     risky_bins,
            "ch1_features":   ch1_features,
            "ch2_features":   ch2_features,
            "explanation":    explanation}

    def run(self, X_train, y_noisy, y_true,
            X_test, y_test,
            noise_rate=0.45, seed=42,
            dataset="unknown",
            X_train_raw=None, X_test_raw=None):

        aug = DataAugmentor(seed=seed)

        # 모든 반환 변수 최상단 초기화
        f1_baseline = 0.0
        f1_ct       = 0.0
        f1_standard = 0.0
        f1_vlm      = 0.0

        def make_rf():
            return RandomForestClassifier(
                n_estimators=500, n_jobs=-1,
                random_state=seed)

        def eval_f1(m, X, y):
            return float(f1_score(
                y, m.predict(X),
                average="macro", zero_division=0))

        # Baseline (raw, 전처리 없음)
        f1_ct = None  # 기본값
        if X_train_raw is not None:
            clf = make_rf()
            clf.fit(X_train_raw, y_noisy)
            f1_baseline = eval_f1(clf, X_test_raw, y_test)
        else:
            clf = make_rf()
            clf.fit(X_train, y_noisy)
            f1_baseline = eval_f1(clf, X_test, y_test)
        f1_ct = f1_baseline  # 기본값 초기화

        # ── CT only (CoTeaching → Aug → RF) ──────
        ct_only = CoTeaching(
            noise_rate=noise_rate, n_rounds=5,
            keep_pct_start=70.0, keep_pct_end=85.0,
            n_estimators=100)
        ct_only.fit_clean(X_train, y_noisy)
        if hasattr(ct_only, '_keep_mask') and \
           ct_only._keep_mask.sum() >= 20:
            km_ct = ct_only._keep_mask
            X_ct_only = X_train[km_ct]
            y_ct_only = y_noisy[km_ct]
            Xa_ct, ya_ct = aug.augment(
                X_ct_only, y_ct_only)
            clf = make_rf()
            clf.fit(Xa_ct, ya_ct)
            f1_ct = eval_f1(clf, X_test, y_test)


        # Step 1: CT filter
        ct = CoTeaching(
            noise_rate=noise_rate, n_rounds=5,
            keep_pct_start=70.0, keep_pct_end=85.0,
            n_estimators=100)
        ct.fit_clean(X_train, y_noisy)
        if not hasattr(ct, '_keep_mask') or \
           ct._keep_mask.sum() < 20:
            return {"f1_baseline": f1_baseline,
                    "f1_ct":       f1_ct,
                    "f1_standard": f1_baseline,
                    "f1_vlm":      f1_baseline,
                    "vlm_features": [],
                    "vlm_bins": [],
                    "vlm_explanation": "CT 실패"}
        km   = ct._keep_mask
        X_ct = X_train[km]
        y_ct = y_noisy[km]

        # Step 2: density image 생성 (ACD 전)
        print("  density image 생성 중...")
        (combined, be_map, ma_map,
         diff, db) =             self._build_density_image(X_ct, y_ct)
        Image.fromarray(combined).save(IMG_TMP)

        # Step 3: VLM 분석 (ACD 전에 먼저)
        print("  VLM 분석 중...")
        vlm_result     = self._vlm_analyze(
            combined, dataset, noise_rate)
        risky_features = vlm_result["risky_features"]
        risky_bins     = vlm_result["risky_bins"]
        explanation    = vlm_result["explanation"]

        # overlap 기반 VLM 필요 여부 판단
        dimg_check = db.build()
        overlap_total = float(np.minimum(
            dimg_check[:,:,0],
            dimg_check[:,:,1]).sum())
        OVERLAP_THRESH = 30.0
        # VLM 파싱 성공 여부
        vlm_parsed = len(risky_features) > 0 and             any(f < X_ct.shape[1] for f in risky_features)
        # overlap 낮으면 Standard, 높으면 VLM
        use_vlm = (overlap_total >= OVERLAP_THRESH)
        print("  [overlap={:.1f}] use_vlm={}  "
              "vlm_parsed={}".format(
              overlap_total, use_vlm, vlm_parsed))

        # Step 4a: Standard ACD (VLM 없이)
        acd_std = AdaptiveClusterDensity(
            max_k=3, minor_thresh=0.15,
            pure_thresh=0.80, cons_thresh=0.7,
            k_knn=min(15, len(X_ct)//10),
            top_remove_pct=20.0,
            top_flip_pct=10.0,
            seed=seed)
        acd_std.fit(X_ct, y_ct, y_true=y_true[km])
        X_sc, y_sc = acd_std.clean(X_ct, y_ct)
        X_us = X_ct[acd_std.U_mask_]
        y_us = y_ct[acd_std.U_mask_]
        if len(X_us) > 0:
            X_sc = np.vstack([X_sc, X_us])
            y_sc = np.concatenate([y_sc, y_us])

        # ACD 과도 제거 방지: clean 비율 75% 미만이면
        # CT 결과 그대로 사용 (ACD 건너뜀)
        clean_ratio = len(X_sc) / max(len(X_ct), 1)
        if clean_ratio < 0.75:
            print("  [ACD skip] clean ratio={:.2f} "
                  "< 0.75 → CT 결과 사용".format(
                  clean_ratio))
            X_sc = X_ct.copy()
            y_sc = y_ct.copy()
        # 방법 E: density feature 추가 (use_vlm일 때만)
        def add_density_feat(X, be_d, ma_d):
            """feature별 be/ma density + entropy 추가"""
            be_f = be_d.mean(axis=0)
            ma_f = ma_d.mean(axis=0)
            total = be_f + ma_f + 1e-8
            p1 = be_f / total; p2 = ma_f / total
            ent_f = -(p1*np.log2(p1+1e-9) +
                      p2*np.log2(p2+1e-9))
            extra = np.column_stack(
                [be_f, ma_f, ent_f]).flatten()
            extra_tiled = np.tile(extra, (len(X), 1))
            return np.hstack([X, extra_tiled])

        if use_vlm:
            X_sc_e  = add_density_feat(
                X_sc, be_map, ma_map)
            X_test_e = add_density_feat(
                X_test, be_map, ma_map)
        else:
            X_sc_e   = X_sc
            X_test_e = X_test

        Xa_s, ya_s = aug.augment(X_sc_e, y_sc)
        clf = make_rf(); clf.fit(Xa_s, ya_s)
        f1_standard = eval_f1(clf, X_test_e, y_test)

        # Step 4b: VLM boost ACD
        acd_vlm = AdaptiveClusterDensity(
            max_k=3, minor_thresh=0.15,
            pure_thresh=0.80, cons_thresh=0.7,
            k_knn=min(15, len(X_ct)//10),
            top_remove_pct=20.0,
            top_flip_pct=0.0,
            seed=seed)
        acd_vlm.fit(X_ct, y_ct, y_true=y_true[km])

        # density에서 risky 구간 추출
        dimg  = db.build()
        be_d  = dimg[:, :, 0]
        ma_d  = dimg[:, :, 1]
        edges = db.bin_edges
        n_b   = dimg.shape[0]
        overlap_per_feat = np.minimum(be_d, ma_d).sum(axis=0)

        valid_vlm = [f for f in risky_features
                     if f < X_ct.shape[1] and
                     overlap_per_feat[f] > 1e-6]
        if not valid_vlm:
            valid_vlm = np.argsort(
                overlap_per_feat)[-5:].tolist()
            valid_vlm = [f for f in valid_vlm
                         if overlap_per_feat[f] > 1e-6]

        src = "VLM" if any(
            f in risky_features for f in valid_vlm
        ) else "auto"

        # risky_mask 생성
        risky_mask = np.zeros(len(X_ct), dtype=bool)
        if valid_vlm:
            for f in valid_vlm:
                try:
                    e = edges[f]
                    b_idxs = np.clip(
                        np.searchsorted(
                            e, X_ct[:, f], "right")-1,
                        0, n_b-1).astype(int)
                    ov = np.array([
                        np.minimum(be_d, ma_d)[b, f]
                        for b in b_idxs])
                    ov_nz = ov[ov > 0]
                    if len(ov_nz) > 0:
                        thresh = np.percentile(ov_nz, 90)
                        risky_mask |= (ov >= thresh)
                except:
                    continue

        # ACD clean
        X_vc, y_vc = acd_vlm.clean(X_ct, y_ct)

        # VLM ACD 과도 제거 방지
        clean_ratio_vlm = len(X_vc) / max(len(X_ct), 1)
        if clean_ratio_vlm < 0.75:
            print("  [VLM ACD skip] ratio={:.2f} "
                  "→ CT 결과 사용".format(clean_ratio_vlm))
            X_vc = X_ct.copy()
            y_vc = y_ct.copy()

        # Selective Augmentation
        if use_vlm and risky_mask.sum() > 0:
            kept_ct_idx = np.where(
                acd_vlm._keep_mask)[0]
            risky_vc_idx = np.array([
                i for i, ct_i in
                enumerate(kept_ct_idx)
                if ct_i < len(risky_mask) and
                risky_mask[ct_i]])
            if len(risky_vc_idx) > 5:
                X_risky = X_vc[risky_vc_idx]                     if len(risky_vc_idx) <= len(X_vc)                     else X_vc
                y_risky = y_vc[risky_vc_idx]                     if len(risky_vc_idx) <= len(y_vc)                     else y_vc
                if (y_risky==0).sum() > 2 and                    (y_risky==1).sum() > 2:
                    X_ra, y_ra = aug.augment(
                        X_risky, y_risky)
                    X_vc = np.vstack([X_vc, X_ra])
                    y_vc = np.concatenate([y_vc, y_ra])
                    print("  [VLM Sel-Aug({})] "
                          "risky={} +{}개".format(
                          src, len(risky_vc_idx),
                          len(X_ra)))

        # U cluster 합류
        X_uv = X_ct[acd_vlm.U_mask_]
        y_uv = y_ct[acd_vlm.U_mask_]
        if len(X_uv) > 0:
            X_vc = np.vstack([X_vc, X_uv])
            y_vc = np.concatenate([y_vc, y_uv])

        # 방법 D+E: density feature + noisy_score 추가
        if use_vlm:
            # X_vc 길이에 맞게 noisy_score 생성
            ns_vc = np.zeros(len(X_vc))
            kept_idx = np.where(acd_vlm._keep_mask)[0]
            ns_from_acd = acd_vlm.noisy_score[
                acd_vlm._keep_mask]
            # kept 샘플만 score 할당 (U cluster는 0)
            ns_vc[:len(ns_from_acd)] = ns_from_acd
            X_vc_e = add_density_feat(X_vc, be_map, ma_map)
            X_vc_e = np.hstack([X_vc_e, ns_vc.reshape(-1,1)])
            X_test_e2 = add_density_feat(X_test, be_map, ma_map)
            X_test_e2 = np.hstack([
                X_test_e2, np.zeros((len(X_test),1))])
        else:
            X_vc_e   = X_vc
            X_test_e2 = X_test

        if (y_vc==0).sum() > 5 and            (y_vc==1).sum() > 5:
            Xa_v, ya_v = aug.augment(X_vc_e, y_vc)
            clf = make_rf(); clf.fit(Xa_v, ya_v)
            f1_vlm = eval_f1(clf, X_test_e2, y_test)
        else:
            f1_vlm = f1_standard

        return {
            "f1_baseline":     f1_baseline,
            "f1_ct":           f1_ct,
            "f1_standard":     f1_standard,
            "f1_vlm":          f1_vlm,
            "vlm_features":    risky_features,
            "vlm_bins":        risky_bins,
            "vlm_explanation": explanation,
            "density_image":   IMG_TMP}
