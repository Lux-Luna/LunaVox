import onnxruntime as ort
import numpy as np
from typing import List, Optional
import threading

from ..Audio.ReferenceAudio import ReferenceAudio
from ..Japanese.JapaneseG2P import japanese_to_phones
from ..English.EnglishG2P import english_to_phones
from ..Chinese.ChineseG2P import chinese_clean_g2p_and_norm
from ..Chinese.ZhBert import compute_bert_phone_features
from ..Utils.Constants import BERT_FEATURE_DIM


class LunaVoxEngine:
    def __init__(self):
        self.stop_event: threading.Event = threading.Event()

    def tts(
            self,
            text: str,
            prompt_audio: ReferenceAudio,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            vocoder: ort.InferenceSession,
            language: str = "ja",
    ) -> Optional[np.ndarray]:
        if language == "en":
            ids = english_to_phones(text)
            text_seq: np.ndarray = np.array([ids], dtype=np.int64)
            text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
        elif language == "zh":
            ids, word2ph, norm_text = chinese_clean_g2p_and_norm(text)
            text_seq: np.ndarray = np.array([ids], dtype=np.int64)
            # Full zh-BERT parity: compute 1024-d features and align to phones
            bert_phone = compute_bert_phone_features(norm_text, word2ph)  # (len_phones, 1024)
            if bert_phone.shape[0] != text_seq.shape[1]:
                text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
            else:
                text_bert = bert_phone
        else:
            text_seq: np.ndarray = np.array([japanese_to_phones(text)], dtype=np.int64)
            text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
        ref_seq = prompt_audio.phonemes_seq
        if ref_seq is None:
            return None
        ref_bert = prompt_audio.text_bert
        if ref_bert is None or ref_bert.shape[0] != ref_seq.shape[1]:
            ref_bert = np.zeros((ref_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)

        semantic_tokens: np.ndarray = self.t2s_cpu(
            ref_seq=ref_seq,
            ref_bert=ref_bert,
            text_seq=text_seq,
            text_bert=text_bert,
            ssl_content=prompt_audio.ssl_content,
            encoder=encoder,
            first_stage_decoder=first_stage_decoder,
            stage_decoder=stage_decoder,
        )
        if self.stop_event.is_set():
            return None

        eos_indices = np.where(semantic_tokens >= 1024)  # 剔除不合法的元素，例如 EOS Token。
        if len(eos_indices[0]) > 0:
            first_eos_index = eos_indices[-1][0]
            semantic_tokens = semantic_tokens[..., :first_eos_index]

        audio_32k = np.expand_dims(prompt_audio.audio_32k, axis=0)  # 增加 Batch_Size 维度
        return vocoder.run(None, {
            "text_seq": text_seq,
            "pred_semantic": semantic_tokens,
            "ref_audio": audio_32k
        })[0]

    def t2s_cpu(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
    ) -> Optional[np.ndarray]:
        """在CPU上运行T2S模型"""
        # Encoder
        x, prompts = encoder.run(
            None,
            {
                "ref_seq": ref_seq,
                "text_seq": text_seq,
                "ref_bert": ref_bert,
                "text_bert": text_bert,
                "ssl_content": ssl_content,
            },
        )
        # First Stage Decoder
        fs_outputs = first_stage_decoder.run(None, {"x": x, "prompts": prompts})
        fs_out_info = first_stage_decoder.get_outputs()
        fs_out_names: List[str] = [o.name for o in fs_out_info]

        # Expected (variant A): aggregated outputs [y, k, v, y_emb, x_example]
        def _fs_get(name: str, default_idx: int):
            if name in fs_out_names:
                return fs_outputs[fs_out_names.index(name)]
            if default_idx < len(fs_outputs):
                return fs_outputs[default_idx]
            return None

        # Variant B: per-layer caches 'present_k_layer_i'/'present_v_layer_i'
        def _collect_layers(prefix: str):
            layers = []
            for idx, nm in enumerate(fs_out_names):
                if nm.startswith(prefix):
                    try:
                        li = int(nm.split("_layer_")[-1])
                    except Exception:
                        li = idx
                    layers.append((li, fs_outputs[idx]))
            layers.sort(key=lambda x: x[0])
            return [arr for _, arr in layers]

        y = _fs_get("y", 0)
        k_agg = _fs_get("k", 1)
        v_agg = _fs_get("v", 2)
        y_emb = _fs_get("y_emb", 3)
        x_example = _fs_get("x_example", 4)
        k_layers = _collect_layers("present_k_layer_")
        v_layers = _collect_layers("present_v_layer_")
        if not k_layers:
            k_layers = None
        if not v_layers:
            v_layers = None

        # Stage Decoder
        stage_in_info = stage_decoder.get_inputs()
        stage_in_names: List[str] = [i.name for i in stage_in_info]
        stage_out_info = stage_decoder.get_outputs()
        stage_out_names: List[str] = [o.name for o in stage_out_info]

        # Determine number of per-layer cache inputs expected
        n_past_k = sum(1 for n in stage_in_names if n.startswith("past_k_layer_"))
        n_past_v = sum(1 for n in stage_in_names if n.startswith("past_v_layer_"))
        n_layers = max(n_past_k, n_past_v)

        # If stage expects per-layer caches but only aggregated provided, try to split along axis 0
        if n_layers > 0 and k_layers is None and k_agg is not None:
            try:
                k_layers = list(np.split(k_agg, n_layers, axis=0))
            except Exception:
                k_layers = None
        if n_layers > 0 and v_layers is None and v_agg is not None:
            try:
                v_layers = list(np.split(v_agg, n_layers, axis=0))
            except Exception:
                v_layers = None

        def _build_stage_feed(_y, _y_emb, _k_layers, _v_layers, _k_agg, _v_agg, _x_example):
            feed = {}
            for name in stage_in_names:
                if name == "iy":
                    feed[name] = _y
                elif name == "iy_emb":
                    feed[name] = _y_emb
                elif name == "ix_example" and _x_example is not None:
                    feed[name] = _x_example
                elif name == "ik" and _k_agg is not None:
                    feed[name] = _k_agg
                elif name == "iv" and _v_agg is not None:
                    feed[name] = _v_agg
                elif name.startswith("past_k_layer_") and _k_layers is not None:
                    try:
                        li = int(name.split("_layer_")[-1])
                        if 0 <= li < len(_k_layers):
                            feed[name] = _k_layers[li]
                    except Exception:
                        pass
                elif name.startswith("past_v_layer_") and _v_layers is not None:
                    try:
                        li = int(name.split("_layer_")[-1])
                        if 0 <= li < len(_v_layers):
                            feed[name] = _v_layers[li]
                    except Exception:
                        pass
            return feed

        def _unpack_stage_outputs(outputs_list, prev_y_emb):
            out_map = {name: outputs_list[idx] for idx, name in enumerate(stage_out_names)}
            _y = out_map.get("y", outputs_list[0] if outputs_list else None)
            _y_emb = out_map.get("y_emb", prev_y_emb)
            _k_agg = out_map.get("k", None)
            _v_agg = out_map.get("v", None)
            # per-layer presents
            pres_k_layers = []
            pres_v_layers = []
            for nm, arr in out_map.items():
                if nm.startswith("present_k_layer_"):
                    try:
                        li = int(nm.split("_layer_")[-1])
                    except Exception:
                        li = 0
                    pres_k_layers.append((li, arr))
                elif nm.startswith("present_v_layer_"):
                    try:
                        li = int(nm.split("_layer_")[-1])
                    except Exception:
                        li = 0
                    pres_v_layers.append((li, arr))
            pres_k_layers = [a for _, a in sorted(pres_k_layers, key=lambda x: x[0])] if pres_k_layers else None
            pres_v_layers = [a for _, a in sorted(pres_v_layers, key=lambda x: x[0])] if pres_v_layers else None
            _logits = out_map.get("logits", None)
            _samples = out_map.get("samples", None)
            return _y, _y_emb, pres_k_layers, pres_v_layers, _k_agg, _v_agg, _logits, _samples

        idx: int = 0
        for idx in range(0, 500):
            if self.stop_event.is_set():
                return None

            input_feed = _build_stage_feed(y, y_emb, k_layers, v_layers, k_agg, v_agg, x_example)
            outputs_list = stage_decoder.run(None, input_feed)
            y, y_emb, new_k_layers, new_v_layers, new_k_agg, new_v_agg, logits, samples = _unpack_stage_outputs(outputs_list, y_emb)

            # Update caches for next step
            if new_k_layers is not None and new_v_layers is not None:
                k_layers, v_layers = new_k_layers, new_v_layers
                k_agg, v_agg = None, None
            else:
                k_agg, v_agg = new_k_agg if new_k_agg is not None else k_agg, new_v_agg if new_v_agg is not None else v_agg

            # EOS/停机判定：优先使用 samples，其次用 logits argmax，最后用 y 值范围
            stop = False
            if samples is not None:
                try:
                    val = int(samples.flat[0])
                    if val >= 1024:
                        stop = True
                except Exception:
                    pass
            elif logits is not None:
                try:
                    last = logits[..., -1, :]
                    val = int(np.argmax(last))
                    if val >= 1024:
                        stop = True
                except Exception:
                    pass
            else:
                try:
                    if int(y.flat[-1]) >= 1024:
                        stop = True
                except Exception:
                    pass

            if stop:
                break

        y[0, -1] = 0
        return np.expand_dims(y[:, -idx:], axis=0)


tts_client: LunaVoxEngine = LunaVoxEngine()
