import json
"""
파인튜닝 데이터셋 구성
density image (CH1, CH2, CH3_differential) + instruction-answer 쌍

입력: LabelDiffusion 실험 결과 (npz)
출력: VLM 파인튜닝용 JSON (image + QA)
"""

import sys, os, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde

sys.path.insert(0, "/home/oper/LabelDiffusion")
from models.density_map    import DensityMapBuilder
from models.baselines      import CoTeaching
from models.augmentation   import DataAugmentor

DOH_TRAIN  = "/home/oper/DoHBrw2020/npz/doh_train.npz"
DOH_TEST   = "/home/oper/DoHBrw2020/npz/doh_test.npz"
GOTHAM_NPZ = "/home/oper/LabelDiffusion/data/gotham_seqs_pcap.npz"
OUT_DIR    = "/home/oper/VLM_IDS/data"
IMG_DIR    = "/home/oper/VLM_IDS/data/images"
os.makedirs(IMG_DIR, exist_ok=True)

# --- 누락된 상수 정의 ---
DOH_QUESTIONS = [
    "이 DoH 트래픽 density map에서 CH2(malicious)가 고밀도인 high bin 구간을 찾고, 그 구간에 benign 샘플이 혼재하는지 분석해줘.",
    "DNS tunneling 트래픽은 패킷이 크기 때문에 high bin에 집중돼. 이 density map에서 CH2 고밀도 구간과 noisy label 위치를 찾아줘.",
    "이 density map에서 malicious(CH2)로 레이블됐지만 benign(CH1) 영역에 있는 noisy 샘플 비율을 추정해줘.",
    "이 DoH 네트워크 트래픽 density map을 분석해서 noisy label 탐지 결과를 요약해줘. CH1은 benign, CH2는 malicious야.",
    "이미지 상의 밀도 분포를 보고, DoH 공격 패턴과 노이즈 레이블의 상관관계를 설명해줘."
]

GOTHAM_QUESTIONS = [
    "이 Gotham IoT 트래픽 density map에서 CH1(benign)과 CH2(malicious)가 가장 밀집된 feature 구간을 찾고, noisy label이 집중된 영역을 설명해줘.",
    "CH3(differential map)에서 경계가 불명확한 구간(값이 0에 가까운 곳)이 어디인지 찾고, 이 구간의 noisy label 위험도를 평가해줘.",
    "IoT flooding 트래픽은 패킷이 작고 IAT가 짧아서 low bin에 집중돼. 이 density map에서 benign과 malicious 경계 구간을 찾아줘.",
    "이 Gotham 네트워크 트래픽 density map을 분석해서 noisy label 탐지 결과를 요약해줘. CH1은 benign, CH2는 malicious야.",
    "전체적인 밀도 지도를 기반으로 Gotham 데이터셋의 노이즈 특성을 분석해줘."
]

# --- 누락된 이미지 증강 함수 정의 ---
def augment_image(img_arr, aug_type):
    """VLM 학습을 위한 간단한 이미지 증강"""
    img = Image.fromarray(img_arr)
    if aug_type == 1:
        # 밝기 조절
        from PIL import ImageEnhance
        img = ImageEnhance.Brightness(img).enhance(1.2)
    elif aug_type == 2:
        # 좌우 반전 (좌표 기반 질문이 아닐 경우 유용)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_type == 3:
        # 약간의 노이즈 추가 또는 대비 조절
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(1.1)
    return np.array(img)


def add_noise(labels, p, seed):
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    noisy[rng.random(len(labels)) < p] ^= 1
    return noisy


def build_density_image(X, y, n_bins=50,
                         img_size=224):
    """
    3채널 density image 생성
    CH1(R): benign density (log 강조)
    CH2(G): malicious density (log 강조)
    CH3(B): Shannon Entropy Map (혼재 구간 밝게)
    """
    db = DensityMapBuilder(
        n_bins=n_bins, cons_thresh=0.7,
        k=min(20, len(X)//4))
    db.fit(X, y)
    dimg   = db.build()
    be_map = dimg[:, :, 0]
    ma_map = dimg[:, :, 1]

    # CH1, CH2: log 스케일 강조
    be_log  = np.log1p(be_map * 10)
    ma_log  = np.log1p(ma_map * 10)
    be_norm = np.clip(
        be_log / (be_log.max() + 1e-8), 0, 1)
    ma_norm = np.clip(
        ma_log / (ma_log.max() + 1e-8), 0, 1)

    # CH3: Shannon Entropy Map
    total   = be_map + ma_map + 1e-8
    p1      = be_map / total
    p2      = ma_map / total
    entropy = -(
        p1 * np.log2(p1 + 1e-9) +
        p2 * np.log2(p2 + 1e-9))
    density_mask = np.tanh(total * 5)
    conflict     = entropy * density_mask
    diff         = be_map - ma_map  # 기존 호환성 유지
    conflict_norm = np.clip(
        conflict / (conflict.max() + 1e-8), 0, 1)

    def to_img(arr, size=img_size, gamma=1.2):
        # contrast stretch: 상위 2% 기준 정규화
        vmax = np.percentile(arr[arr > 0], 98)             if (arr > 0).any() else 1.0
        arr = np.clip(arr / (vmax + 1e-8), 0, 1)
        arr = np.power(arr, 1.0/gamma)
        img = Image.fromarray(
            (arr * 255).astype(np.uint8))
        return np.array(img.resize(
            (size, size), Image.LANCZOS))

    combined = np.stack([
        to_img(be_norm),
        to_img(ma_norm),
        to_img(conflict_norm)
    ], axis=-1).astype(np.uint8)

    return combined, be_map, ma_map, diff


def compute_noisy_score(X, y, dimg_be, dimg_ma, n_bins=50):
    """density image로 per-sample noisy score 계산"""
    be_idx = np.where(y==0)[0]
    ma_idx = np.where(y==1)[0]
    if len(be_idx)<5 or len(ma_idx)<5:
        return np.zeros(len(X))

    noisy = np.zeros(len(X), dtype=np.float32)
    cnt   = 0
    for f in range(X.shape[1]):
        try:
            b = X[be_idx, f]; m = X[ma_idx, f]
            if b.std()<1e-6 or m.std()<1e-6: continue
            kbd = gaussian_kde(b); kmd = gaussian_kde(m)
            bd  = kbd(X[:, f]); md  = kmd(X[:, f])
            ratio = bd / (bd + md + 1e-8)
            noisy += np.abs(
                ratio - (1 - y.astype(float)))
            cnt += 1
        except: continue
    if cnt > 0: noisy /= cnt
    return noisy


def extract_doh_features(be_map, ma_map, diff,
                          y_noisy, noisy_score,
                          y_true=None, X=None):
    """
    DoH 특화 feature 추출
    실제 noisy label 위치 기반으로 risky_features 계산
    """
    n_bins = be_map.shape[0]

    # 실제 noisy 샘플 기반 risky_features (방법 1)
    if y_true is not None and X is not None:
        noisy_mask = (y_noisy != y_true)
        if noisy_mask.sum() > 3 and X.shape[1] > 0:
            # noisy 샘플과 clean 샘플의 feature 분포 차이
            X_noisy = X[noisy_mask]
            X_clean = X[~noisy_mask]
            # feature별 noisy/clean 분포 차이 (KS-like)
            feat_diff = np.abs(
                X_noisy.mean(axis=0) -
                X_clean.mean(axis=0)) / (
                X.std(axis=0) + 1e-8)
            risky_feats = np.argsort(
                feat_diff)[-3:].tolist()
        else:
            risky_feats = None
    else:
        risky_feats = None

    # density map 기반 (fallback)
    ma_bin_sum  = ma_map.sum(axis=0)
    ma_top_bin  = np.argsort(ma_bin_sum)[-5:]
    ma_feat_sum = ma_map.sum(axis=1)
    ma_top_feat = np.argsort(ma_feat_sum)[-3:]

    overlap_by_bin = np.minimum(be_map, ma_map).sum(axis=0)
    noisy_bin = np.argsort(overlap_by_bin)[-5:]

    # overlap 기반 risky_feat (fallback)
    overlap_by_feat = np.minimum(be_map, ma_map).sum(axis=1)
    overlap_top_feat = np.argsort(overlap_by_feat)[-3:]

    # overlap 기반 상위 feature
    overlap_by_feat = np.minimum(
        be_map, ma_map).sum(axis=1)
    overlap_top2 = np.argsort(
        overlap_by_feat)[-2:].tolist()

    # 최종: noisy 3개 + overlap 2개 결합
    if risky_feats:
        combined = list(set(risky_feats) |
                        set(overlap_top2))
    else:
        combined = sorted(overlap_top_feat.tolist())
    final_risky = sorted(combined)

    be_noisy = noisy_score[y_noisy==0]
    ma_noisy = noisy_score[y_noisy==1]
    be_noisy_rate = (be_noisy >= np.percentile(
        noisy_score, 80)).mean() * 100         if len(be_noisy) > 0 else 0
    ma_noisy_rate = (ma_noisy >= np.percentile(
        noisy_score, 80)).mean() * 100         if len(ma_noisy) > 0 else 0

    separation = "명확" if         np.minimum(be_map, ma_map).sum() < 0.1 else         "중간" if         np.minimum(be_map, ma_map).sum() < 0.3 else         "불명확"

    return {
        "risky_feat":    final_risky,
        "ma_top_bin":    sorted(ma_top_bin.tolist()),
        "ma_top_feat":   sorted(ma_top_feat.tolist()),
        "noisy_bin":     sorted(noisy_bin.tolist()),
        "be_top_feat":   sorted(
            np.argsort(be_map.sum(axis=1))[-3:].tolist()),
        "be_top_bin":    sorted(
            np.argsort(be_map.sum(axis=0))[-3:].tolist()),
        "be_noisy_rate": be_noisy_rate,
        "ma_noisy_rate": ma_noisy_rate,
        "separation":    separation,
        "overlap":       float(
            np.minimum(be_map, ma_map).sum())
    }


def extract_gotham_features(be_map, ma_map, diff,
                             y_noisy, noisy_score,
                             y_true=None, X=None):
    """
    Gotham 특화 feature 추출
    실제 noisy label 위치 기반으로 risky_features 계산
    """
    n_bins = be_map.shape[0]

    # 실제 noisy 샘플 기반 risky_features (방법 1)
    if y_true is not None and X is not None:
        noisy_mask = (y_noisy != y_true)
        if noisy_mask.sum() > 3 and X.shape[1] > 0:
            X_noisy = X[noisy_mask]
            X_clean = X[~noisy_mask]
            feat_diff = np.abs(
                X_noisy.mean(axis=0) -
                X_clean.mean(axis=0)) / (
                X.std(axis=0) + 1e-8)
            risky_feats = np.argsort(
                feat_diff)[-3:].tolist()
        else:
            risky_feats = None
    else:
        risky_feats = None

    # CH3 경계 구간 (diff ≈ 0)
    abs_diff      = np.abs(diff)
    boundary_feat = np.argsort(
        abs_diff.min(axis=1))[:3]
    boundary_bin  = np.argsort(
        abs_diff.min(axis=0))[:3]

    # overlap 기반 fallback
    overlap_by_feat = np.minimum(
        be_map, ma_map).sum(axis=1)
    overlap_top_feat = np.argsort(
        overlap_by_feat)[-3:]

    # overlap 기반 상위 feature
    overlap_by_feat = np.minimum(
        be_map, ma_map).sum(axis=1)
    overlap_top2 = np.argsort(
        overlap_by_feat)[-2:].tolist()

    # 최종: noisy 3개 + overlap 2개 결합
    if risky_feats:
        combined = list(set(risky_feats) |
                        set(overlap_top2))
    else:
        combined = sorted(boundary_feat.tolist())
    final_risky = sorted(combined)

    be_noisy = noisy_score[y_noisy==0]
    ma_noisy = noisy_score[y_noisy==1]
    be_noisy_rate = (be_noisy >= np.percentile(
        noisy_score, 80)).mean() * 100         if len(be_noisy) > 0 else 0
    ma_noisy_rate = (ma_noisy >= np.percentile(
        noisy_score, 80)).mean() * 100         if len(ma_noisy) > 0 else 0

    separation = "명확" if         np.minimum(be_map, ma_map).sum() < 0.1 else         "중간" if         np.minimum(be_map, ma_map).sum() < 0.3 else         "불명확"

    return {
        "risky_feat":    final_risky,
        "be_top_feat":   sorted(
            np.argsort(be_map.sum(axis=1))[-3:].tolist()),
        "be_top_bin":    sorted(
            np.argsort(be_map.sum(axis=0))[-3:].tolist()),
        "ma_top_feat":   sorted(
            np.argsort(ma_map.sum(axis=1))[-3:].tolist()),
        "ma_top_bin":    sorted(
            np.argsort(ma_map.sum(axis=0))[-3:].tolist()),
        "boundary_feat": sorted(boundary_feat.tolist()),
        "boundary_bin":  sorted(boundary_bin.tolist()),
        "be_noisy_rate": be_noisy_rate,
        "ma_noisy_rate": ma_noisy_rate,
        "separation":    separation,
        "overlap":       float(
            np.minimum(be_map, ma_map).sum())
    }


def extract_image_features(be_map, ma_map, diff,
                            y_noisy, noisy_score):
    """
    density image에서 실제 시각적 특징 추출
    → QA 답변에 이미지 특징 반영
    """
    n_bins = be_map.shape[0]

    # CH1: benign 밀도 집중 구간
    be_feat_sum = be_map.sum(axis=1)  # feature별 합산
    be_top_feat = np.argsort(be_feat_sum)[-3:]
    be_bin_sum  = be_map.sum(axis=0)  # bin별 합산
    be_top_bin  = np.argsort(be_bin_sum)[-3:]

    # CH2: malicious 밀도 집중 구간
    ma_feat_sum = ma_map.sum(axis=1)
    ma_top_feat = np.argsort(ma_feat_sum)[-3:]
    ma_bin_sum  = ma_map.sum(axis=0)
    ma_top_bin  = np.argsort(ma_bin_sum)[-3:]

    # CH3: 경계 영역 (diff ≈ 0인 구간)
    abs_diff    = np.abs(diff)
    boundary_feat = np.argsort(
        abs_diff.min(axis=1))[:3]  # diff 작은 = 경계
    boundary_bin  = np.argsort(
        abs_diff.min(axis=0))[:3]

    # noisy score 높은 클래스별 비율
    be_noisy = noisy_score[y_noisy==0]
    ma_noisy = noisy_score[y_noisy==1]
    be_noisy_rate = (
        be_noisy >= np.percentile(
            noisy_score, 80)).mean() * 100
    ma_noisy_rate = (
        ma_noisy >= np.percentile(
            noisy_score, 80)).mean() * 100

    # 분리도: be/ma density 겹치는 정도
    overlap = np.minimum(be_map, ma_map).sum()
    separation = "명확" if overlap < 0.1 else                  "중간" if overlap < 0.3 else "불명확"

    return {
        "be_top_feat": sorted(be_top_feat.tolist()),
        "be_top_bin":  sorted(be_top_bin.tolist()),
        "ma_top_feat": sorted(ma_top_feat.tolist()),
        "ma_top_bin":  sorted(ma_top_bin.tolist()),
        "boundary_feat": sorted(boundary_feat.tolist()),
        "boundary_bin":  sorted(boundary_bin.tolist()),
        "be_noisy_rate": be_noisy_rate,
        "ma_noisy_rate": ma_noisy_rate,
        "separation":    separation,
        "overlap":       float(overlap)
    }


def make_qa_pair(noisy_score, y_noisy, y_true,
                 dataset, p, n_sample,
                 be_map=None, ma_map=None, diff=None):
    """
    데이터셋별 특화 QA 쌍 생성
    DoH:    CH2 high bin 패턴 타겟
    Gotham: CH1/CH2 경계 low bin 패턴 타겟
    """
    actual_noisy_rate = (y_noisy != y_true).mean()
    is_doh = "DoH" in dataset

    # 데이터셋별 특화 feature 추출
    if be_map is not None:
        if is_doh:
            feat = extract_doh_features(
                be_map, ma_map, diff,
                y_noisy, noisy_score)
        else:
            feat = extract_gotham_features(
                be_map, ma_map, diff,
                y_noisy, noisy_score)
    else:
        feat = None

    # 데이터셋별 다른 질문
    rng = np.random.default_rng(
        int(p*100) + n_sample + hash(dataset) % 1000)
    q_type = rng.integers(0, 4)

    if is_doh:
        if q_type == 0:
            question = (
                "이 DoH 트래픽 density map에서 "
                "CH2(malicious)가 고밀도인 "
                "high bin 구간을 찾고, "
                "그 구간에 benign 샘플이 "
                "혼재하는지 분석해줘.")
        elif q_type == 1:
            question = (
                "DNS tunneling 트래픽은 "
                "패킷이 크기 때문에 high bin에 "
                "집중돼. 이 density map에서 "
                "CH2 고밀도 구간과 noisy label "
                "위치를 찾아줘.")
        elif q_type == 2:
            question = (
                "이 density map에서 "
                "malicious(CH2)로 레이블됐지만 "
                "benign(CH1) 영역에 있는 "
                "noisy 샘플 비율을 추정해줘.")
        else:
            question = (
                "이 DoH 네트워크 트래픽 "
                "density map을 분석해서 "
                "noisy label 탐지 결과를 요약해줘. "
                "CH1은 benign, CH2는 malicious야.")
    else:
        if q_type == 0:
            question = (
                "이 Gotham IoT 트래픽 density map에서 "
                "CH1(benign)과 CH2(malicious)가 "
                "가장 밀집된 feature 구간을 찾고, "
                "noisy label이 집중된 영역을 설명해줘.")
        elif q_type == 1:
            question = (
                "CH3(differential map)에서 "
                "경계가 불명확한 구간(값이 0에 가까운 곳)이 "
                "어디인지 찾고, 이 구간의 noisy label "
                "위험도를 평가해줘.")
        elif q_type == 2:
            question = (
                "IoT flooding 트래픽은 "
                "패킷이 작고 IAT가 짧아서 "
                "low bin에 집중돼. 이 density map에서 "
                "benign과 malicious 경계 구간을 찾아줘.")
        else:
            question = (
                "이 Gotham 네트워크 트래픽 "
                "density map을 분석해서 "
                "noisy label 탐지 결과를 요약해줘. "
                "CH1은 benign, CH2는 malicious야.")

    # 데이터셋별 특화 답변
    if feat is not None:
        if is_doh:
            parts = [
                "[density map 분석 결과]",
                "데이터셋: {} / 노이즈율: {:.0f}% / N={}".format(
                    dataset, p*100, n_sample),
                "",
                "CH2(malicious) 고밀도 구간:",
                "  feature: {} / bin: {}".format(
                    feat["ma_top_feat"],
                    feat["ma_top_bin"]),
                "",
                "CH1(benign) 집중 구간:",
                "  feature: {} / bin: {}".format(
                    feat["be_top_feat"],
                    feat["be_top_bin"]),
                "",
                "noisy 위험 bin 구간 (CH1+CH2 혼재):",
                "  bin: {}".format(feat["noisy_bin"]),
                "",
                "be/ma 분리도: {} (overlap={:.3f})".format(
                    feat["separation"], feat["overlap"]),
                "noisy 의심: benign {:.1f}% / "
                "malicious {:.1f}%".format(
                    feat["be_noisy_rate"],
                    feat["ma_noisy_rate"]),
                "실제 noisy label 비율: {:.1f}%".format(
                    actual_noisy_rate*100),
                "",
                "판단: DoH tunneling 패턴 - "
                "CH2 고밀도 bin {} 구간에서 "
                "noisy 집중. 상위 20% 제거 권장.".format(
                    feat["noisy_bin"])
            ]
        else:
            parts = [
                "[density map 분석 결과]",
                "데이터셋: {} / 노이즈율: {:.0f}% / N={}".format(
                    dataset, p*100, n_sample),
                "",
                "CH1(benign) 집중 구간:",
                "  feature: {} / bin: {}".format(
                    feat["be_top_feat"],
                    feat["be_top_bin"]),
                "",
                "CH2(malicious) 집중 구간:",
                "  feature: {} / bin: {}".format(
                    feat["ma_top_feat"],
                    feat["ma_top_bin"]),
                "",
                "CH3 경계 불명확 구간 (noisy label 위험 높음):",
                "  feature: {} / bin: {}".format(
                    feat["boundary_feat"],
                    feat["boundary_bin"]),
                "",
                "be/ma 분리도: {} (overlap={:.3f})".format(
                    feat["separation"], feat["overlap"]),
                "noisy 의심: benign {:.1f}% / "
                "malicious {:.1f}%".format(
                    feat["be_noisy_rate"],
                    feat["ma_noisy_rate"]),
                "실제 noisy label 비율: {:.1f}%".format(
                    actual_noisy_rate*100),
                "",
                "판단: Gotham IoT 패턴 - "
                "feature {}-{} 경계 구간에서 "
                "noisy 집중. 상위 20% 제거 권장.".format(
                    min(feat["boundary_feat"]),
                    max(feat["boundary_feat"]))
            ]
        answer = "\n".join(parts)
    else:
        answer = "분석 실패"

    questions = DOH_QUESTIONS if is_doh else GOTHAM_QUESTIONS
    return questions, feat, actual_noisy_rate, is_doh


def build_answer(feat, is_doh, anr, name, p, N):
    """자연어 분석 + JSON 요약 동시 출력"""
    if feat is None:
        return "분석 실패\nRESULT: {}"
    if is_doh:
        # noisy 위험 구간: CH1+CH2 혼재 bin
        risky_f = sorted(set(
            feat.get("ma_top_feat",[]) +
            feat.get("be_top_feat",[])))[:5]
        risky_b = feat.get("noisy_bin", [])
        parts = [
            "[density map 분석 결과]",
            "데이터셋: {} / 노이즈율: {:.0f}% / N={}".format(
                name, p*100, N),
            "",
            "CH2(malicious) 고밀도 구간:",
            "  feature: {} / bin: {}".format(
                feat.get("ma_top_feat",""),
                feat.get("ma_top_bin","")),
            "",
            "CH1(benign) 집중 구간:",
            "  feature: {} / bin: {}".format(
                feat.get("be_top_feat",""),
                feat.get("be_top_bin","")),
            "",
            "noisy 위험 bin 구간 (CH1+CH2 혼재):",
            "  bin: {}".format(feat.get("noisy_bin","")),
            "",
            "be/ma 분리도: {} (overlap={:.3f})".format(
                feat.get("separation",""),
                feat.get("overlap",0)),
            "noisy 의심: benign {:.1f}% / "
            "malicious {:.1f}%".format(
                feat.get("be_noisy_rate",0),
                feat.get("ma_noisy_rate",0)),
            "실제 noisy label 비율: {:.1f}%".format(
                anr*100),
            "",
            "판단: DoH tunneling - CH2 고밀도 bin {} "
            "구간 noisy 집중. 상위 20% 제거 권장.".format(
                feat.get("noisy_bin","")),
            "",
            "RESULT: {}".format(json.dumps({
                "risky_features": feat.get("risky_feat", risky_f),
                "risky_bins": risky_b,
                "reason": "DoH noisy 샘플 집중 feature"
            }, ensure_ascii=False))
        ]
    else:
        bf = feat.get("boundary_feat", [])
        risky_f = bf if bf else feat.get("be_top_feat",[])
        risky_b = feat.get("boundary_bin", [])
        parts = [
            "[density map 분석 결과]",
            "데이터셋: {} / 노이즈율: {:.0f}% / N={}".format(
                name, p*100, N),
            "",
            "CH1(benign) 집중 구간:",
            "  feature: {} / bin: {}".format(
                feat.get("be_top_feat",""),
                feat.get("be_top_bin","")),
            "",
            "CH2(malicious) 집중 구간:",
            "  feature: {} / bin: {}".format(
                feat.get("ma_top_feat",""),
                feat.get("ma_top_bin","")),
            "",
            "CH3 경계 불명확 구간 (noisy 위험):",
            "  feature: {} / bin: {}".format(
                feat.get("boundary_feat",""),
                feat.get("boundary_bin","")),
            "",
            "be/ma 분리도: {} (overlap={:.3f})".format(
                feat.get("separation",""),
                feat.get("overlap",0)),
            "noisy 의심: benign {:.1f}% / "
            "malicious {:.1f}%".format(
                feat.get("be_noisy_rate",0),
                feat.get("ma_noisy_rate",0)),
            "실제 noisy label 비율: {:.1f}%".format(
                anr*100),
            "",
            "판단: Gotham IoT - feature {} 경계 구간 "
            "noisy 집중. 상위 20% 제거 권장.".format(
                risky_f),
            "",
            "RESULT: {}".format(json.dumps({
                "risky_features": feat.get("risky_feat", risky_f),
                "risky_bins": risky_b,
                "reason": "Gotham noisy 샘플 집중 feature"
            }, ensure_ascii=False))
        ]
    return "\n".join(parts)


def subsample_data(Xl, Xi, Xa, y_true, N, seed):
    """N=1000 데이터에서 N개 랜덤 subsample"""
    rng = np.random.default_rng(seed + 1000)
    be  = np.where(y_true==0)[0]
    ma  = np.where(y_true==1)[0]
    n2  = min(N//2, len(be), len(ma))
    idx = np.concatenate([
        rng.choice(be, n2, replace=False),
        rng.choice(ma, n2, replace=False)])
    rng.shuffle(idx)
    return Xl[idx], Xi[idx], Xa[idx], y_true[idx]


def process_dataset(name, load_fn,
                    noise_levels, n_sizes, seeds):
    records = []
    for N in n_sizes:
        for p in noise_levels:
            for seed in seeds:
                try:
                    # 데이터 로드
                    Xl, Xi, Xa, y_true, \
                        Xl_te, y_te = load_fn(seed, N)
                    y_noisy = add_noise(y_true, p, seed)

                    # CT filter
                    sc  = StandardScaler()
                    X_s = sc.fit_transform(Xl).astype(
                        np.float32)
                    ct  = CoTeaching(
                        noise_rate=p, n_rounds=5,
                        keep_pct_start=70.0,
                        keep_pct_end=85.0,
                        n_estimators=100)
                    ct.fit_clean(X_s, y_noisy)
                    if not hasattr(ct, '_keep_mask'):
                        continue
                    km = ct._keep_mask
                    if km.sum() < 20: continue
                    X_ct = X_s[km]; y_ct = y_noisy[km]

                    # density image 생성
                    img_arr, be_map, ma_map, diff = \
                        build_density_image(X_ct, y_ct)

                    # noisy score 계산
                    ns = compute_noisy_score(
                        X_ct, y_ct, be_map, ma_map)

                    # 이미지 저장
                    fname = "{}_p{:03d}_N{}_s{}.png".format(
                        name, int(p*100), N, seed)
                    fpath = os.path.join(IMG_DIR, fname)
                    # img_arr이 이미 224×224×3
                    Image.fromarray(img_arr).save(fpath)

                    # QA 생성 (5개 질문 × aug)
                    questions, feat, anr, is_doh =                         make_qa_pair(
                            ns, y_ct, y_true[km],
                            name, p, N,
                            be_map=be_map,
                            ma_map=ma_map,
                            diff=diff)

                    answer = build_answer(
                        feat, is_doh, anr, name, p, N)

                    # 5개 질문 × 원본 이미지
                    for qi, q in enumerate(questions):
                        records.append({
                            "image":    fpath,
                            "dataset":  name,
                            "noise_rate": p,
                            "n_sample": N,
                            "seed":     seed,
                            "question": q,
                            "answer":   answer,
                            "noisy_rate_actual": float(
                                (y_ct !=
                                 y_true[km]).mean())
                        })

                    # 3가지 augmentation 이미지
                    for aug_t in [1, 2, 3]:
                        try:
                            aug_arr = augment_image(
                                img_arr, aug_t)
                            aug_fname =                                 "{}_p{:03d}_N{}_s{}"                                 "_aug{}.png".format(
                                name, int(p*100),
                                N, seed, aug_t)
                            aug_fpath = os.path.join(
                                IMG_DIR, aug_fname)
                            Image.fromarray(
                                aug_arr).save(aug_fpath)
                            # aug 이미지당 2개 질문
                            for q in questions[:2]:
                                records.append({
                                    "image":   aug_fpath,
                                    "dataset": name,
                                    "noise_rate": p,
                                    "n_sample": N,
                                    "seed":    seed,
                                    "question": q,
                                    "answer":  answer,
                                    "noisy_rate_actual":
                                        float((y_ct !=
                                            y_true[km]
                                        ).mean())
                                })
                        except: pass

                    print("  {} p={} N={} s={} "
                          "→ {}개 QA".format(
                          name, p, N, seed,
                          len(questions) + 6))

                    # window sliding: N=1000에서 추가 subsample
                    if N == 1000:
                        for ws in range(1, 5):
                            try:
                                Xl_w, Xi_w, Xa_w, y_w =                                     subsample_data(
                                        Xl, Xi, Xa,
                                        y_true, 500,
                                        seed*100+ws)
                                y_nw = add_noise(y_w, p,
                                    seed*100+ws)
                                sc2  = StandardScaler()
                                Xw_s = sc2.fit_transform(
                                    Xl_w).astype(np.float32)
                                ct2  = CoTeaching(
                                    noise_rate=p,
                                    n_rounds=5,
                                    keep_pct_start=70.0,
                                    keep_pct_end=85.0,
                                    n_estimators=100)
                                ct2.fit_clean(Xw_s, y_nw)
                                if not hasattr(
                                        ct2, "_keep_mask"):
                                    continue
                                km2 = ct2._keep_mask
                                if km2.sum() < 20: continue
                                X_w2 = Xw_s[km2]
                                y_w2 = y_nw[km2]
                                img2, bm2, mm2, df2 =                                     build_density_image(
                                        X_w2, y_w2)
                                ns2 = compute_noisy_score(
                                    X_w2, y_w2, bm2, mm2)
                                fn2 = "{}_p{:03d}_N{}_s{}_w{}.png".format(
                                    name, int(p*100),
                                    N, seed, ws)
                                fp2 = os.path.join(
                                    IMG_DIR, fn2)
                                Image.fromarray(img2).save(fp2)
                                result2 = make_qa_pair(
                                    ns2, y_w2,
                                    y_w[km2], name, p, 500,
                                    be_map=bm2,
                                    ma_map=mm2,
                                    diff=df2,
                                    X=X_w2)
                                qs2, ft2, anr2, isd2 = result2
                                a2 = build_answer(
                                    ft2, isd2,
                                    anr2, name, p, 500)
                                for q2 in qs2[:3]:
                                    records.append({
                                        "image": fp2,
                                        "dataset": name,
                                        "noise_rate": p,
                                        "n_sample": 500,
                                        "seed": seed*100+ws,
                                        "question": q2,
                                        "answer": a2,
                                        "noisy_rate_actual":
                                        float((y_w2 !=
                                            y_w[km2]).mean())
                                    })
                            except: continue

                except Exception as e:
                    import traceback
                    print("  오류: {}".format(e))
                    traceback.print_exc()
    return records


def load_doh(seed, n_sample):
    tr  = np.load(DOH_TRAIN)
    te  = np.load(DOH_TEST)
    rng = np.random.default_rng(seed)
    def sample(d, n):
        Xl = d['seqs_len'].astype(np.float32)
        Xi = d['seqs_iat'].astype(np.float32)
        Xs = d['seqs_signed'].astype(np.float32)
        y  = d['labels'].astype(int)
        be = rng.permutation(np.where(y==0)[0])
        ma = rng.permutation(np.where(y==1)[0])
        n2 = min(n//2, len(be), len(ma))
        idx = np.concatenate([be[:n2], ma[:n2]])
        rng.shuffle(idx)
        return Xl[idx], Xi[idx], Xs[idx], y[idx]
    Xl_tr,Xi_tr,Xs_tr,y_tr = sample(tr, n_sample)
    Xl_te,Xi_te,Xs_te,y_te = sample(te, 4000)
    return Xl_tr,Xi_tr,Xs_tr,y_tr,Xl_te,y_te


def load_gotham(seed, n_sample):
    d   = np.load(GOTHAM_NPZ, allow_pickle=True)
    Xl  = d['seqs_len'].astype(np.float32)
    Xi  = d['seqs_iat'].astype(np.float32)
    Xf  = d['seqs_flags'].astype(np.float32)
    y   = d['labels'].astype(int)
    rng = np.random.default_rng(seed)
    be  = rng.permutation(np.where(y==0)[0])
    ma  = rng.permutation(np.where(y==1)[0])
    n2  = min(n_sample//2, len(be), len(ma))
    tr  = np.concatenate([be[:n2], ma[:n2]])
    rng.shuffle(tr)
    n_te = min(2000, len(be)-n2, len(ma)-n2)
    te   = np.concatenate([
        be[n2:n2+n_te], ma[n2:n2+n_te]])
    return Xl[tr],Xi[tr],Xf[tr],y[tr],Xl[te],y[te]


if __name__ == "__main__":
    NOISE_LEVELS = [0.20, 0.25, 0.30, 0.35,
                    0.40, 0.45]
    N_SIZES      = [250, 500, 1000]
    SEEDS        = list(range(200))  # 200 seeds

    all_records = []

    print("=== DoH2020 ===")
    all_records += process_dataset(
        "DoH2020", load_doh,
        NOISE_LEVELS, N_SIZES, SEEDS)

    print("=== Gotham2025 ===")
    all_records += process_dataset(
        "Gotham2025", load_gotham,
        NOISE_LEVELS, N_SIZES, SEEDS)

    # train/val split (9:1)
    np.random.shuffle(all_records)
    n_val  = max(10, len(all_records)//10)
    val    = all_records[:n_val]
    train  = all_records[n_val:]

    with open(os.path.join(
            OUT_DIR, "train.json"), "w") as f:
        json.dump(train, f, indent=2,
                  ensure_ascii=False)
    with open(os.path.join(
            OUT_DIR, "val.json"), "w") as f:
        json.dump(val, f, indent=2,
                  ensure_ascii=False)

    print("\n총 {}개 샘플".format(len(all_records)))
    print("train: {}  val: {}".format(
        len(train), len(val)))
    print("이미지 저장:", IMG_DIR)
